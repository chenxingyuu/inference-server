#include "publisher/RedisPublisher.h"
#include "common/Logger.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

#ifdef BUILD_REDIS_PUBLISHER
#include <sw/redis++/redis++.h>

namespace infer {
namespace {
// Production redis client wrapping redis-plus-plus.
class RedisPlusPlusClient final : public IRedisStreamsClient {
public:
    explicit RedisPlusPlusClient(const RedisConfig& cfg) {
        sw::redis::ConnectionOptions opts;
        opts.host = cfg.host;
        opts.port = cfg.port;
        redis_ = std::make_unique<sw::redis::Redis>(opts);
    }

    void xadd(const std::string& key,
              const std::vector<std::pair<std::string,std::string>>& fields,
              long long maxlen) override {
        redis_->xadd(key, "*", fields.begin(), fields.end(),
                     sw::redis::MAXLEN{maxlen, false});
    }

private:
    std::unique_ptr<sw::redis::Redis> redis_;
};
} // namespace
} // namespace infer
#endif // BUILD_REDIS_PUBLISHER

namespace infer {

RedisPublisher::RedisPublisher(const RedisConfig& cfg)
    : cfg_(cfg) {
#ifdef BUILD_REDIS_PUBLISHER
    client_ = std::make_unique<RedisPlusPlusClient>(cfg);
#else
    throw std::runtime_error(
        "RedisPublisher: server was built without BUILD_REDIS_PUBLISHER=ON");
#endif
    thread_ = std::thread(&RedisPublisher::publishLoop, this);
}

RedisPublisher::RedisPublisher(const RedisConfig& cfg,
                               std::unique_ptr<IRedisStreamsClient> client)
    : cfg_(cfg), client_(std::move(client)) {
    thread_ = std::thread(&RedisPublisher::publishLoop, this);
}

RedisPublisher::~RedisPublisher() {
    stop_flag_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void RedisPublisher::publish(InferResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (static_cast<int>(queue_.size()) >= cfg_.queue_capacity) {
        ++dropped_;
        LOG_WARN("RedisPublisher: queue full, dropping result for stream {}",
                 result.stream_id);
        return;
    }
    queue_.push(std::move(result));
    cv_.notify_one();
}

void RedisPublisher::flush() {
    // Wait until the queue drains.
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return queue_.empty() || stop_flag_.load(); });
}

void RedisPublisher::publishLoop() {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_.load(); });

        while (!queue_.empty()) {
            InferResult r = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            cv_.notify_all();  // wake flush() waiter if queue just became empty

            try {
                const std::string key = streamKey(r.stream_id);
                const std::string data = serialize(r);
                client_->xadd(key, {{"data", data}}, cfg_.max_len);
                ++published_;
            } catch (const std::exception& e) {
                LOG_ERROR("RedisPublisher: xadd failed: {}", e.what());
            }

            lock.lock();
        }
        cv_.notify_all();  // notify flush() that queue is empty

        if (stop_flag_.load() && queue_.empty()) break;
    }
}

std::string RedisPublisher::streamKey(const std::string& stream_id) const {
    return cfg_.stream_prefix + ":" + stream_id;
}

std::string RedisPublisher::serialize(const InferResult& r) const {
    nlohmann::json j;
    j["stream_id"]  = r.stream_id;
    j["frame_ts"]   = r.frame_ts;
    j["infer_ts"]   = r.infer_ts;
    j["latency_ms"] = r.latency_ms;
    j["model_id"]   = r.model_id;
    j["frame_seq"]  = r.frame_seq;
    j["detections"] = nlohmann::json::array();
    for (const auto& d : r.detections) {
        nlohmann::json det;
        det["class_id"]   = d.class_id;
        det["class_name"] = d.class_name;
        det["confidence"] = d.confidence;
        det["bbox"]       = {d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2};
        if (d.track_id.has_value()) det["track_id"] = d.track_id.value();
        det["attributes"] = d.attributes;
        j["detections"].push_back(std::move(det));
    }
    return j.dump();
}

} // namespace infer
