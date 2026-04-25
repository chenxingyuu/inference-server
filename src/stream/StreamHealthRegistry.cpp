#include "stream/StreamHealthRegistry.h"
#include "common/Logger.h"
#include <algorithm>
#include <chrono>
#include <mutex>

namespace infer {

StreamHealthRegistry& StreamHealthRegistry::get() {
    static StreamHealthRegistry instance;
    return instance;
}

double StreamHealthRegistry::now() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        system_clock::now().time_since_epoch()).count();
}

void StreamHealthRegistry::onStreamAdded(const std::string& id, int degraded_threshold, int max_reconnect_attempts) {
    std::unique_lock lock(mu_);
    // Entry contains std::atomic members so it is not movable; construct in-place.
    auto& entry = map_[id];
    entry.health.state            = StreamState::CONNECTING;
    entry.health.state_changed_at = now();
    entry.degraded_threshold      = degraded_threshold;
    entry.max_reconnect_attempts  = max_reconnect_attempts;
    entry.frame_ts_atomic.store(0.0, std::memory_order_relaxed);
    entry.frame_count_atomic.store(0, std::memory_order_relaxed);
}

void StreamHealthRegistry::onStreamOpened(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    if (h.state == StreamState::FAILED) return;
    // First open: CONNECTING→STREAMING. Subsequent opens handled via onReconnectSucceeded.
    if (h.state == StreamState::CONNECTING) {
        h.state            = StreamState::STREAMING;
        h.state_changed_at = now();
        LOG_INFO("[{}] stream state: CONNECTING -> STREAMING", id);
    }
}

void StreamHealthRegistry::onStreamDropped(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    if (h.state == StreamState::FAILED) return;
    if (h.state == StreamState::STREAMING || h.state == StreamState::CONNECTING) {
        LOG_WARN("[{}] stream state: {} -> RECONNECTING",
                 id, streamStateStr(h.state));
        h.state            = StreamState::RECONNECTING;
        h.state_changed_at = now();
    }
}

void StreamHealthRegistry::onReconnectFailed(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& entry = *it;
    auto& h     = entry.second.health;
    if (h.state == StreamState::FAILED) return;
    ++h.consecutive_failures;
    const uint32_t max_attempts = static_cast<uint32_t>(std::max(0, entry.second.max_reconnect_attempts));
    if (max_attempts > 0 && h.consecutive_failures >= max_attempts) {
        LOG_ERROR("[{}] stream state: {} -> FAILED (max_reconnect_attempts={} reached)",
                  entry.first, streamStateStr(h.state), max_attempts);
        h.state            = StreamState::FAILED;
        h.state_changed_at = now();
        return;
    }
    if (h.consecutive_failures >= static_cast<uint32_t>(entry.second.degraded_threshold)
        && h.state != StreamState::DEGRADED) {
        LOG_WARN("[{}] stream state: RECONNECTING -> DEGRADED (consecutive_failures={})",
                 entry.first, h.consecutive_failures);
        h.state            = StreamState::DEGRADED;
        h.state_changed_at = now();
    }
}

void StreamHealthRegistry::onReconnectSucceeded(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    if (h.state == StreamState::FAILED) return;
    LOG_INFO("[{}] stream state: {} -> STREAMING (reconnect_count={})",
             id, streamStateStr(h.state), h.reconnect_count + 1);
    h.state               = StreamState::STREAMING;
    h.state_changed_at    = now();
    h.consecutive_failures = 0;
    ++h.reconnect_count;
}

void StreamHealthRegistry::onFrameDecoded(const std::string& id, double capture_ts) {
    // Use shared_lock: we only need the map structure to be stable (no insert/erase),
    // not exclusive ownership, because the hot-path fields are atomics.
    std::shared_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    it->second.frame_ts_atomic.store(capture_ts, std::memory_order_relaxed);
    it->second.frame_count_atomic.fetch_add(1, std::memory_order_relaxed);
}

void StreamHealthRegistry::onStreamRemoved(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    if (h.state == StreamState::FAILED) return;
    h.state            = StreamState::STOPPED;
    h.state_changed_at = now();
}

StreamHealth StreamHealthRegistry::getHealth(const std::string& id) const {
    std::shared_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) {
        StreamHealth h;
        h.state = StreamState::STOPPED;
        return h;
    }
    StreamHealth h = it->second.health;
    h.last_frame_ts        = it->second.frame_ts_atomic.load(std::memory_order_relaxed);
    h.frames_since_last_hb = it->second.frame_count_atomic.load(std::memory_order_relaxed);
    return h;
}

std::vector<std::pair<std::string, StreamHealth>> StreamHealthRegistry::getAllHealth() const {
    std::shared_lock lock(mu_);
    std::vector<std::pair<std::string, StreamHealth>> result;
    result.reserve(map_.size());
    for (const auto& [id, entry] : map_) {
        StreamHealth h = entry.health;
        h.last_frame_ts        = entry.frame_ts_atomic.load(std::memory_order_relaxed);
        h.frames_since_last_hb = entry.frame_count_atomic.load(std::memory_order_relaxed);
        result.emplace_back(id, h);
    }
    return result;
}

void StreamHealthRegistry::resetHbCounters() {
    // Only atomics are modified; shared_lock is sufficient to stabilize the map.
    std::shared_lock lock(mu_);
    for (auto& [id, entry] : map_)
        entry.frame_count_atomic.store(0, std::memory_order_relaxed);
}

void StreamHealthRegistry::clear() {
    std::unique_lock lock(mu_);
    map_.clear();
}

} // namespace infer
