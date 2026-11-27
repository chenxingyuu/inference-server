#include "pipeline/TaskManager.h"
#include "common/Logger.h"
#include "stream/StreamHealthRegistry.h"

namespace infer {
namespace {

using namespace std::chrono;
constexpr std::size_t kMaxConcurrentProbes = 2;

bool supportsProbe(const PipelineSourceConfig& source) {
    return source.url.compare(0, 7, "rtsp://") == 0;
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

void TaskManager::trackSourceProbeLocked(const PipelineSourceConfig& source) {
    if (supportsProbe(source)) {
        source_probes_.emplace(source.id, std::make_shared<SourceProbeRecord>());
    }
}

void TaskManager::startSourceMonitoring(SourceProbe probe, milliseconds interval) {
    if (!probe || !source_probe_workers_.empty()) return;
    {
        std::lock_guard lock(mu_);
        source_probe_exit_ = false;
        source_probe_interval_ = interval;
    }
    try {
        for (std::size_t i = 0; i < kMaxConcurrentProbes; ++i) {
            source_probe_workers_.emplace_back([this, probe] { watchSources(probe); });
        }
    } catch (...) {
        stopSourceMonitoring();
        throw;
    }
}

void TaskManager::stopSourceMonitoring() {
    {
        std::lock_guard lock(mu_);
        source_probe_exit_ = true;
    }
    source_probe_cv_.notify_all();
    for (auto& worker : source_probe_workers_) worker.join();
    source_probe_workers_.clear();
}

void TaskManager::watchSources(const SourceProbe& probe) {
    std::unique_lock lock(mu_);
    while (!source_probe_exit_) {
        if (active_source_probes_.size() >= kMaxConcurrentProbes) {
            source_probe_cv_.wait(lock);
            continue;
        }

        const PipelineSourceConfig* next = nullptr;
        auto due = steady_clock::time_point::max();
        for (const auto& source : cfg_.sources) {
            const auto record = source_probes_.find(source.id);
            if (record == source_probes_.end() || active_source_probes_.count(source.id)) continue;
            if (record->second->next_check < due) {
                next = &source;
                due = record->second->next_check;
            }
        }
        if (!next) {
            source_probe_cv_.wait(lock);
        } else if (due > steady_clock::now()) {
            source_probe_cv_.wait_until(lock, due);
        } else {
            runSourceProbeLocked(*next, probe, lock);
        }
    }
}

SourceInfo TaskManager::sourceInfoLocked(const PipelineSourceConfig& source) const {
    const auto legacy = StreamHealthRegistry::get().getHealth(source.id);
    SourceInfo info{source.id, source.url, streamStateStr(legacy.state),
                    static_cast<int>(legacy.reconnect_count)};
    const auto probe = source_probes_.find(source.id);
    if (probe == source_probes_.end()) {
        info.availability.status = "unsupported";
    } else if (probe->second->result) {
        const auto& last = *probe->second->result;
        info.last_probe = last;
        info.availability = {last.passed ? "online" : "offline", "probe",
                             last.checked_at_ms, last.reason};
    }
    return info;
}

SourceProbeResponse TaskManager::probeSource(const std::string& id, const SourceProbe& probe) {
    std::unique_lock lock(mu_);
    const auto* source = cfg_.findSource(id);
    if (!source) return {{}, false, "source_not_found"};
    if (!supportsProbe(*source)) return {{}, false, "source_probe_unsupported"};
    if (!probe || source_probe_exit_) return {{}, false, "source_probe_unavailable"};
    if (active_source_probes_.count(id)) return {{}, false, "source_probe_in_progress"};
    if (active_source_probes_.size() >= kMaxConcurrentProbes) return {{}, false, "source_probe_busy"};
    return runSourceProbeLocked(*source, probe, lock);
}

SourceProbeResponse TaskManager::runSourceProbeLocked(const PipelineSourceConfig& source,
                                                      const SourceProbe& probe,
                                                      std::unique_lock<std::mutex>& lock) {
    const auto config = source;
    const auto record = source_probes_.at(config.id);
    active_source_probes_.insert(config.id);
    lock.unlock();

    const auto started = steady_clock::now();
    auto code = SourceProbeCode::Unavailable;
    try {
        code = probe(config);
    } catch (...) {
        // Exceptions from FFmpeg or process setup may contain connection credentials.
    }
    const auto checked = steady_clock::now();
    const auto timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    const auto elapsed = duration_cast<milliseconds>(checked - started).count();
    const auto* reason = probeReason(code);

    lock.lock();
    active_source_probes_.erase(config.id);
    source_probe_cv_.notify_all();
    const auto found = source_probes_.find(config.id);
    if (found == source_probes_.end() || found->second != record) {
        return {{}, false, "source_not_found"};
    }

    record->next_check = checked + source_probe_interval_;
    if (!reason) {
        LOG_WARN("[{}] source probe unavailable ({}ms)", config.id, elapsed);
        return {{}, false, "source_probe_unavailable"};
    }
    const bool passed = code == SourceProbeCode::Passed;
    if (!record->result || record->result->passed != passed || record->result->reason != reason) {
        LOG_INFO("[{}] source probe {} ({}ms)", config.id, passed ? "passed" : reason, elapsed);
    }
    record->result = SourceProbeResult{passed, timestamp, elapsed, reason};
    return {sourceInfoLocked(*cfg_.findSource(config.id)), false, {}};
}

} // namespace infer
