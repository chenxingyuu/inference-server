#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <chrono>
#include <cstdint>
#include <atomic>

namespace infer {

enum class StreamState {
    CONNECTING   = 0,
    STREAMING    = 1,
    RECONNECTING = 2,
    DEGRADED     = 3,
    FAILED       = 4,
    STOPPED      = 5,
};

// Convert StreamState to string for JSON / logging
inline const char* streamStateStr(StreamState s) {
    switch (s) {
        case StreamState::CONNECTING:   return "CONNECTING";
        case StreamState::STREAMING:    return "STREAMING";
        case StreamState::RECONNECTING: return "RECONNECTING";
        case StreamState::DEGRADED:     return "DEGRADED";
        case StreamState::FAILED:       return "FAILED";
        case StreamState::STOPPED:      return "STOPPED";
    }
    return "UNKNOWN";
}

struct StreamHealth {
    StreamState state{StreamState::CONNECTING};
    double      last_frame_ts{0.0};         // epoch seconds of most recent decoded frame
    double      state_changed_at{0.0};      // epoch seconds when current state was entered
    uint32_t    reconnect_count{0};         // total reconnect successes (monotonic)
    uint32_t    consecutive_failures{0};    // consecutive open failures (cleared on success)
    uint64_t    frames_since_last_hb{0};    // frames decoded since last heartbeat reset
};

// Thread-safe singleton registry tracking per-stream health state.
// FFmpegDecoder calls the on*() methods; HeartbeatPublisher and Metrics read via get*().
class StreamHealthRegistry {
public:
    static StreamHealthRegistry& get();

    // Called by StreamPool::addStream before the decoder thread starts
    void onStreamAdded(const std::string& id, int degraded_threshold = 5, int max_reconnect_attempts = 0);

    // Called by FFmpegDecoder when openStream() succeeds for the first time
    void onStreamOpened(const std::string& id);

    // Called by FFmpegDecoder when av_read_frame returns an error (stream dropped)
    void onStreamDropped(const std::string& id);

    // Called by FFmpegDecoder when a reconnect attempt (openStream) fails
    // Transitions to DEGRADED when consecutive_failures >= degraded_threshold
    void onReconnectFailed(const std::string& id);

    // Called by FFmpegDecoder when a reconnect attempt (openStream) succeeds
    void onReconnectSucceeded(const std::string& id);

    // Called by FFmpegDecoder on every successfully decoded frame
    void onFrameDecoded(const std::string& id, double capture_ts);

    // Called by StreamPool::removeStream after the decoder thread stops
    void onStreamRemoved(const std::string& id);

    // --- Readers ---

    // Returns a copy of the health snapshot; returns default-constructed StreamHealth
    // with state=STOPPED if id is not found.
    StreamHealth getHealth(const std::string& id) const;

    // Returns all stream health snapshots (copies)
    std::vector<std::pair<std::string, StreamHealth>> getAllHealth() const;

    // Called by HeartbeatPublisher after emitting a heartbeat round;
    // resets frames_since_last_hb to 0 for all streams.
    void resetHbCounters();

    // For testing: clear all entries
    void clear();

private:
    StreamHealthRegistry() = default;

    struct Entry {
        StreamHealth health;
        int          degraded_threshold{5};
        int          max_reconnect_attempts{0};
        std::atomic<double> last_frame_ts_atomic{0.0};
        std::atomic<uint64_t> frames_since_last_hb_atomic{0};
    };

    static double now();

    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, Entry> map_;
};

} // namespace infer
