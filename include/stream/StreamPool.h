#pragma once

#include "stream/FFmpegDecoder.h"
#include "stream/FrameBuffer.h"
#include "common/Config.h"
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace infer {

// Manages a pool of FFmpegDecoder instances (one per camera stream).
// Each decoder pushes frames into a per-stream FrameBuffer.
class StreamPool {
public:
    explicit StreamPool(int max_streams = 100);
    ~StreamPool();

    // Start all streams defined in config
    void start(const AppConfig& cfg);

    // Add a single stream at runtime
    void addStream(const StreamConfig& cfg);

    // Remove a stream at runtime
    void removeStream(const std::string& stream_id);

    // Stop all streams
    void stopAll();

    // Returns nullptr if not found
    FrameBuffer* getBuffer(const std::string& stream_id);

    // Snapshot of all active stream IDs
    std::vector<std::string> activeStreams() const;

    int maxStreams() const noexcept { return max_streams_; }

private:
    struct Entry {
        std::unique_ptr<FFmpegDecoder> decoder;
        std::unique_ptr<FrameBuffer>   buffer;
    };

    int max_streams_;
    mutable std::shared_mutex           mutex_;
    std::unordered_map<std::string, Entry> streams_;
};

} // namespace infer
