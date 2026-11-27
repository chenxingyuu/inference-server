#include "pipeline/TaskManager.h"
#include "common/Logger.h"
#include "stream/StreamHealthRegistry.h"

#include <algorithm>

namespace infer {
namespace {

using namespace std::chrono;

bool supportsProbe(const PipelineSourceConfig& source) {
    return source.url.compare(0, 7, "rtsp://") == 0;
}

SourceAvailability observe(const SourceObservation::Snapshot& snapshot,
                           const PipelineSourceConfig& source, steady_clock::time_point now) {
    using Phase = SourceObservation::Phase;
    if (snapshot.phase == Phase::Reconnecting) {
        return {"reconnecting", "runtime", snapshot.changed.epoch_ms, {}};
    }
    if (snapshot.phase == Phase::Connecting) {
        const auto open_window = milliseconds(std::max<int64_t>(15000, 2LL * source.open_timeout_ms));
        if (now - snapshot.changed.steady > open_window) {
            return {"unavailable", "runtime", snapshot.changed.epoch_ms, "connection_timeout"};
        }
        return {"connecting", "runtime", snapshot.changed.epoch_ms, {}};
    }

    const auto frame_window = milliseconds(std::max(15000, source.read_timeout_ms));
    const auto last_event = snapshot.last_frame.value_or(snapshot.changed);
    if (now - last_event.steady > frame_window) {
        return {"unavailable", "runtime", last_event.epoch_ms,
                snapshot.last_frame ? "no_recent_frame" : "first_frame_timeout"};
    }
    if (snapshot.last_frame) return {"online", "runtime", last_event.epoch_ms, {}};
    return {snapshot.reconnecting_since ? "reconnecting" : "connecting",
            "runtime", last_event.epoch_ms, {}};
}

int priority(const std::string& status) {
    if (status == "online") return 4;
    if (status == "reconnecting") return 3;
    if (status == "connecting") return 2;
    if (status == "unavailable") return 1;
    return 0;
}

const char* probeReason(SourceProbeCode code) {
    switch (code) {
        case SourceProbeCode::Passed: return "";
        case SourceProbeCode::ConnectionFailed: return "connection_failed";
        case SourceProbeCode::ConnectionTimeout: return "connection_timeout";
        case SourceProbeCode::AuthenticationFailed: return "authentication_failed";
        case SourceProbeCode::NoVideo: return "no_video";
        case SourceProbeCode::FirstFrameTimeout: return "first_frame_timeout";
        case SourceProbeCode::DecodeFailed: return "decode_failed";
        case SourceProbeCode::StreamFailed: return "stream_failed";
        case SourceProbeCode::ProbeTimeout: return "probe_timeout";
        case SourceProbeCode::Unavailable: break;
    }
    return nullptr;
}

} // namespace

std::optional<TaskManager::SourceFailure> TaskManager::sourceFailureOnStartupTimeout(
    const Run& run, const PipelineSourceConfig& source) {
    const auto now = SourceEventTime::now();
    std::optional<SourceFailure> failure;
    for (const auto& [_, observation] : run.source_observations) {
        const auto snapshot = observation->snapshot();
        if (snapshot.phase == SourceObservation::Phase::Idle ||
            snapshot.phase == SourceObservation::Phase::Connecting) continue;
        auto status = observe(snapshot, source, now.steady);
        // A decoder that already produces frames rules out an input outage.
        if (status.status == "online") return std::nullopt;
        auto observed = snapshot.last_frame.value_or(snapshot.changed);
        if (status.status == "reconnecting" && snapshot.reconnecting_since) {
            status.reason = "stream_failed";
            observed = *snapshot.reconnecting_since;
        }
        if (!status.reason.empty() && (!failure || observed.steady > failure->observed.steady)) {
            failure = SourceFailure{observed, std::move(status.reason)};
        }
    }
    return failure;
}

SourceInfo TaskManager::sourceInfoLocked(const PipelineSourceConfig& source) const {
    const auto legacy = StreamHealthRegistry::get().getHealth(source.id);
    SourceInfo info{source.id, source.url, streamStateStr(legacy.state),
                    static_cast<int>(legacy.reconnect_count)};
    if (!supportsProbe(source)) {
        info.availability.status = "unsupported";
        return info;
    }

    const auto now = steady_clock::now();
    steady_clock::time_point newest_runtime_event{};
    const SourceFailure* failure = nullptr;
    for (const auto& task : cfg_.tasks) {
        if (task.source_id != source.id) continue;
        const auto it = entries_.find(task.id);
        if (it == entries_.end() || it->second.removed) continue;
        const auto& entry = it->second;
        if (entry.run && entry.run->stop_requested) continue;
        if (entry.source_failure &&
            (!failure || entry.source_failure->observed.steady > failure->observed.steady)) {
            failure = &*entry.source_failure;
        }
        if (!entry.run || entry.run->cancelled.load()) continue;
        for (const auto& [_, observation] : entry.run->source_observations) {
            const auto snapshot = observation->snapshot();
            if (snapshot.phase == SourceObservation::Phase::Idle) continue;
            auto current = observe(snapshot, source, now);
            const auto rank = priority(current.status);
            const auto current_rank = priority(info.availability.status);
            const auto event = snapshot.last_frame.value_or(snapshot.changed).steady;
            if (rank > current_rank ||
                (rank == current_rank && event > newest_runtime_event)) {
                info.availability = std::move(current);
                newest_runtime_event = event;
            }
        }
    }

    const auto probe = source_probes_.find(source.id);
    if (probe != source_probes_.end()) info.last_probe = probe->second->result;
    if (info.availability.basis == "runtime") return info;
    if (failure && (!info.last_probe || failure->observed.steady > probe->second->checked.steady)) {
        info.availability = {"unavailable", "runtime", failure->observed.epoch_ms, failure->reason};
    } else if (info.last_probe) {
        const auto& last = *info.last_probe;
        const bool expired = now - probe->second->checked.steady > seconds(60);
        info.availability = {expired ? "expired" : (last.passed ? "probe_passed" : "probe_failed"),
                             "probe", last.checked_at_ms, last.reason};
    }
    return info;
}

SourceProbeResponse TaskManager::probeSource(const std::string& id, const SourceProbe& probe) {
    std::unique_lock lock(mu_);
    const auto* source = cfg_.findSource(id);
    if (!source) return {{}, false, "source_not_found"};
    if (!supportsProbe(*source)) return {{}, false, "source_probe_unsupported"};
    auto current = sourceInfoLocked(*source);
    if (current.availability.status == "online") return {std::move(current), true, {}};
    if (!probe) return {{}, false, "source_probe_unavailable"};
    if (active_source_probes_.count(id)) return {{}, false, "source_probe_in_progress"};
    if (active_source_probes_.size() >= 2) return {{}, false, "source_probe_busy"};

    const auto config = *source;
    auto& slot = source_probes_[id];
    if (!slot) slot = std::make_shared<SourceProbeRecord>();
    const auto record = slot;
    active_source_probes_.insert(id);
    lock.unlock();

    const auto started = steady_clock::now();
    auto code = SourceProbeCode::Unavailable;
    try {
        code = probe(config);
    } catch (...) {
        // Exceptions from FFmpeg or process setup may contain connection credentials.
    }
    const auto checked = SourceEventTime::now();
    const auto elapsed = duration_cast<milliseconds>(checked.steady - started).count();
    const auto* reason = probeReason(code);

    lock.lock();
    active_source_probes_.erase(id);
    const auto found = source_probes_.find(id);
    source = cfg_.findSource(id);
    if (!source || found == source_probes_.end() || found->second != record) {
        return {{}, false, "source_not_found"};
    }
    if (!reason) {
        lock.unlock();
        LOG_WARN("[{}] source probe unavailable ({}ms)", id, elapsed);
        return {{}, false, "source_probe_unavailable"};
    }
    const bool passed = code == SourceProbeCode::Passed;
    record->checked = checked;
    record->result = SourceProbeResult{passed, checked.epoch_ms, elapsed, reason};
    auto updated = sourceInfoLocked(*source);
    lock.unlock();
    LOG_INFO("[{}] source probe {} ({}ms)", id, passed ? "passed" : reason, elapsed);
    return {std::move(updated), false, {}};
}

} // namespace infer
