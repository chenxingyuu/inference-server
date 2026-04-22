#include "publisher/KafkaPublisher.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace infer {

using json = nlohmann::json;

KafkaPublisher::KafkaPublisher(const KafkaConfig& cfg) : cfg_(cfg) {
    char errstr[512];

    rd_kafka_conf_t* conf = rd_kafka_conf_new();
    rd_kafka_conf_set(conf, "bootstrap.servers", cfg.brokers.c_str(), errstr, sizeof(errstr));
    rd_kafka_conf_set(conf, "compression.codec", cfg.compression.c_str(), errstr, sizeof(errstr));
    {
        std::string linger = std::to_string(cfg.linger_ms);
        rd_kafka_conf_set(conf, "linger.ms", linger.c_str(), errstr, sizeof(errstr));
    }
    rd_kafka_conf_set_dr_msg_cb(conf, &KafkaPublisher::deliveryCallback);

    producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
    if (!producer_) {
        throw std::runtime_error(std::string("KafkaPublisher: rd_kafka_new failed: ") + errstr);
    }

    topic_ = rd_kafka_topic_new(producer_, cfg.topic.c_str(), nullptr);
    if (!topic_) {
        throw std::runtime_error("KafkaPublisher: rd_kafka_topic_new failed");
    }

    thread_ = std::thread(&KafkaPublisher::publishLoop, this);
    LOG_INFO("KafkaPublisher: connected to {} topic={}", cfg.brokers, cfg.topic);
}

KafkaPublisher::~KafkaPublisher() {
    stop_flag_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    flush();
    if (topic_)    rd_kafka_topic_destroy(topic_);
    if (producer_) rd_kafka_destroy(producer_);
}

void KafkaPublisher::publish(InferResult result) {
    std::unique_lock lock(mutex_);
    if (static_cast<int>(queue_.size()) >= cfg_.queue_capacity) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        Metrics::get().incKafkaDropped();
        LOG_ERROR("KafkaPublisher: queue full, dropping result for stream {}",
                  result.stream_id);
        return;
    }
    queue_.push(std::move(result));
    lock.unlock();
    cv_.notify_one();
}

void KafkaPublisher::flush() {
    if (producer_) rd_kafka_flush(producer_, 5000 /*ms*/);
}

void KafkaPublisher::publishLoop() {
    while (!stop_flag_.load()) {
        InferResult r;
        {
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(50),
                         [this] { return !queue_.empty() || stop_flag_.load(); });
            if (queue_.empty()) {
                // Poll to handle delivery callbacks
                rd_kafka_poll(producer_, 0);
                continue;
            }
            r = std::move(queue_.front());
            queue_.pop();
        }

        std::string payload = serialize(r);
        int ret = rd_kafka_produce(
            topic_,
            RD_KAFKA_PARTITION_UA,
            RD_KAFKA_MSG_F_COPY,
            const_cast<char*>(payload.data()), payload.size(),
            r.stream_id.data(), r.stream_id.size(),
            nullptr);

        if (ret == -1) {
            LOG_ERROR("KafkaPublisher: produce failed: {}",
                      rd_kafka_err2str(rd_kafka_last_error()));
        } else {
            published_.fetch_add(1, std::memory_order_relaxed);
            Metrics::get().incKafkaPublished();
        }
        rd_kafka_poll(producer_, 0);
    }
}

std::string KafkaPublisher::serialize(const InferResult& r) const {
    json dets = json::array();
    for (const auto& d : r.detections) {
        json det = {
            {"class_id",   d.class_id},
            {"class_name", d.class_name},
            {"confidence", d.confidence},
            {"bbox", {
                {"x1", d.bbox.x1}, {"y1", d.bbox.y1},
                {"x2", d.bbox.x2}, {"y2", d.bbox.y2}
            }}
        };
        if (d.track_id.has_value()) {
            det["track_id"] = d.track_id.value();
        }
        dets.push_back(std::move(det));
    }
    json msg = {
        {"stream_id",         r.stream_id},
        {"frame_ts",          r.frame_ts},
        {"infer_ts",          r.infer_ts},
        {"latency_ms",        r.latency_ms},
        {"queue_latency_ms",  r.queue_latency_ms},
        {"infer_ms",          r.infer_ms},
        {"decode_ms",         r.decode_ms},
        {"model_id",          r.model_id},
        {"detections",        dets},
        {"frame_local_path",  r.frame_local_path},
        {"frame_url",         r.frame_url},
        {"frame_upload_state",r.frame_upload_state}
    };
    return msg.dump();
}

void KafkaPublisher::deliveryCallback(rd_kafka_t* /*rk*/,
                                       const rd_kafka_message_t* msg,
                                       void* /*opaque*/) {
    if (msg->err) {
        LOG_ERROR("KafkaPublisher: delivery failed: {}", rd_kafka_err2str(msg->err));
    }
}

} // namespace infer
