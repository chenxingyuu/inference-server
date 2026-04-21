#include "stream/StreamHealthRegistry.h"
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
    Entry e;
    e.health.state           = StreamState::CONNECTING;
    e.health.state_changed_at = now();
    e.degraded_threshold     = degraded_threshold;
    e.max_reconnect_attempts = max_reconnect_attempts;
    map_[id] = std::move(e);
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
    }
}

void StreamHealthRegistry::onStreamDropped(const std::string& id) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    if (h.state == StreamState::FAILED) return;
    if (h.state == StreamState::STREAMING || h.state == StreamState::CONNECTING) {
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
        h.state            = StreamState::FAILED;
        h.state_changed_at = now();
        return;
    }
    if (h.consecutive_failures >= static_cast<uint32_t>(entry.second.degraded_threshold)
        && h.state != StreamState::DEGRADED) {
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
    h.state               = StreamState::STREAMING;
    h.state_changed_at    = now();
    h.consecutive_failures = 0;
    ++h.reconnect_count;
}

void StreamHealthRegistry::onFrameDecoded(const std::string& id, double capture_ts) {
    std::unique_lock lock(mu_);
    auto it = map_.find(id);
    if (it == map_.end()) return;
    auto& h = it->second.health;
    h.last_frame_ts = capture_ts;
    ++h.frames_since_last_hb;
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
    return it->second.health;
}

std::vector<std::pair<std::string, StreamHealth>> StreamHealthRegistry::getAllHealth() const {
    std::shared_lock lock(mu_);
    std::vector<std::pair<std::string, StreamHealth>> result;
    result.reserve(map_.size());
    for (const auto& [id, entry] : map_)
        result.emplace_back(id, entry.health);
    return result;
}

void StreamHealthRegistry::resetHbCounters() {
    std::unique_lock lock(mu_);
    for (auto& [id, entry] : map_)
        entry.health.frames_since_last_hb = 0;
}

void StreamHealthRegistry::clear() {
    std::unique_lock lock(mu_);
    map_.clear();
}

} // namespace infer
