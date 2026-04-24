#pragma once

#include "publisher/IPublisher.h"
#include "common/Config.h"
#include <librdkafka/rdkafka.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <string>

namespace infer {

// Kafka publisher backed by librdkafka.
// Serialize InferResult → JSON, produce to configured topic.
// A background thread drains the internal queue to decouple inference latency
// from Kafka I/O.
class KafkaPublisher : public IPublisher {
public:
    explicit KafkaPublisher(const KafkaConfig& cfg);
    ~KafkaPublisher() override;

    void publish(InferResult result) override;
    void flush() override;

private:
    void publishLoop();
    std::string serialize(const InferResult& r) const;

    static void deliveryCallback(rd_kafka_t* rk,
                                 const rd_kafka_message_t* msg,
                                 void* opaque);

    KafkaConfig   cfg_;
    rd_kafka_t*   producer_{nullptr};
    rd_kafka_topic_t* topic_{nullptr};

    std::queue<InferResult> queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::thread             thread_;
    std::atomic<bool>       stop_flag_{false};

    std::atomic<uint64_t>   published_{0};
    std::atomic<uint64_t>   dropped_{0};
    std::atomic<uint64_t>   queue_depth_{0};
};

} // namespace infer
