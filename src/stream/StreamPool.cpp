#include "stream/StreamPool.h"
#include "common/Logger.h"
#include <stdexcept>

namespace infer {

StreamPool::StreamPool(int max_streams) : max_streams_(max_streams) {}

StreamPool::~StreamPool() {
    stopAll();
}

void StreamPool::start(const AppConfig& cfg) {
    for (const auto& scfg : cfg.streams) {
        addStream(scfg);
    }
}

void StreamPool::addStream(const StreamConfig& cfg) {
    std::unique_lock lock(mutex_);
    if (static_cast<int>(streams_.size()) >= max_streams_) {
        throw std::runtime_error("StreamPool: max_streams limit reached");
    }
    if (streams_.count(cfg.id)) {
        LOG_WARN("StreamPool: stream {} already exists, skipping", cfg.id);
        return;
    }

    auto& entry = streams_[cfg.id];
    entry.buffer   = std::make_unique<FrameBuffer>(cfg.id);
    entry.decoder  = std::make_unique<FFmpegDecoder>();
    entry.model_id = cfg.model_id;
    entry.tracker  = cfg.tracker;

    // Capture raw pointer before the lock is released
    FrameBuffer* buf = entry.buffer.get();
    entry.decoder->start(cfg, [buf](Frame f) {
        buf->push(std::move(f));
    });

    LOG_INFO("StreamPool: added stream {}", cfg.id);
}

void StreamPool::removeStream(const std::string& stream_id) {
    std::unique_lock lock(mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) return;
    it->second.decoder->stop();
    streams_.erase(it);
    LOG_INFO("StreamPool: removed stream {}", stream_id);
}

void StreamPool::stopAll() {
    std::unique_lock lock(mutex_);
    for (auto& [id, entry] : streams_) {
        entry.decoder->stop();
    }
    streams_.clear();
    LOG_INFO("StreamPool: all streams stopped");
}

FrameBuffer* StreamPool::getBuffer(const std::string& stream_id) {
    std::shared_lock lock(mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) return nullptr;
    return it->second.buffer.get();
}

std::vector<std::string> StreamPool::activeStreams() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(streams_.size());
    for (const auto& [id, _] : streams_) {
        ids.push_back(id);
    }
    return ids;
}

std::string StreamPool::getStreamModelId(const std::string& stream_id) const {
    std::shared_lock lock(mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) return "";
    return it->second.model_id;
}

TrackerType StreamPool::getStreamTrackerType(const std::string& stream_id) const {
    std::shared_lock lock(mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) return TrackerType::None;
    return it->second.tracker;
}

} // namespace infer
