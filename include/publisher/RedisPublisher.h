#pragma once

#include "publisher/IPublisher.h"
#include "publisher/IRedisStreamsClient.h"
#include "common/Config.h"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <string>

namespace infer {

// Redis Streams publisher.
// Serialises InferResult to JSON and calls XADD {prefix}:{stream_id} MAXLEN ~ max_len.
// A background thread drains the internal queue (same pattern as KafkaPublisher).
// Thread-safe: publish() is safe to call from multiple InferWorker threads.
class RedisPublisher final : public IPublisher {
public:
    explicit RedisPublisher(const RedisConfig& cfg);

    // Test constructor: inject a custom client (no real Redis connection needed).
    RedisPublisher(const RedisConfig& cfg, std::unique_ptr<IRedisStreamsClient> client);

    ~RedisPublisher() override;

    RedisPublisher(const RedisPublisher&)            = delete;
    RedisPublisher& operator=(const RedisPublisher&) = delete;

    void publish(InferResult result) override;
    void flush() override;

private:
    void publishLoop();
    std::string serialize(const InferResult& r) const;
    std::string streamKey(const std::string& stream_id) const;

    RedisConfig                       cfg_;
    std::unique_ptr<IRedisStreamsClient> client_;

    std::queue<InferResult>           queue_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    std::thread                       thread_;
    std::atomic<bool>                 stop_flag_{false};

    std::atomic<uint64_t>             published_{0};
    std::atomic<uint64_t>             dropped_{0};
};

} // namespace infer
