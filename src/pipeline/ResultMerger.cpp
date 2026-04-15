#include "pipeline/ResultMerger.h"
#include "common/Logger.h"

namespace infer {

ResultMerger::ResultMerger(IPublisher& publisher, int max_wait_ms)
    : publisher_(publisher)
    , max_wait_ms_(max_wait_ms)
{}

void ResultMerger::registerPrimary(InferResult result, int expected_count) {
    if (expected_count == 0) {
        publisher_.publish(std::move(result));
        return;
    }

    const std::string key = result.stream_id + ":"
                            + std::to_string(result.frame_ts);

    std::lock_guard<std::mutex> lk(mu_);
    PendingEntry entry;
    entry.result   = std::move(result);
    entry.expected = expected_count;
    entry.deadline = std::chrono::steady_clock::now()
                     + std::chrono::milliseconds(max_wait_ms_);
    pending_.emplace(key, std::move(entry));
}

void ResultMerger::addAttribute(const std::string& stream_id,
                                 double             frame_ts,
                                 int                det_idx,
                                 const std::string& attribute_key,
                                 const std::string& attribute_value) {
    // Key must match what registerPrimary() stored: stream_id + ":" + str(frame_ts)
    const std::string key = stream_id + ":" + std::to_string(frame_ts);

    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(key);
    if (it == pending_.end()) {
        LOG_WARN("ResultMerger: no pending entry for key {}", key);
        return;
    }

    auto& entry = it->second;
    if (det_idx >= 0 && det_idx < static_cast<int>(entry.result.detections.size())) {
        entry.result.detections[det_idx].attributes[attribute_key] = attribute_value;
    }
    entry.received++;

    if (entry.received >= entry.expected) {
        publishAndErase(it);
    }
}

void ResultMerger::flushExpired() {
    std::lock_guard<std::mutex> lk(mu_);
    auto now = std::chrono::steady_clock::now();
    for (auto it = pending_.begin(); it != pending_.end(); ) {
        if (now >= it->second.deadline) {
            LOG_DEBUG("ResultMerger: deadline expired for {}, publishing with {}/{} attributes",
                      it->first, it->second.received, it->second.expected);
            publisher_.publish(std::move(it->second.result));
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

void ResultMerger::publishAndErase(
        std::unordered_map<std::string, PendingEntry>::iterator it) {
    // Called under mu_
    publisher_.publish(std::move(it->second.result));
    pending_.erase(it);
}

} // namespace infer
