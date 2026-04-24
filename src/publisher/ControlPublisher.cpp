#include "publisher/ControlPublisher.h"
#include "common/Logger.h"

#include <nlohmann/json.hpp>
#include <librdkafka/rdkafka.h>
#include <stdexcept>

namespace infer {

using json = nlohmann::json;
namespace {
void requireKafkaConfSet(rd_kafka_conf_t* conf,
                         const char* key,
                         const std::string& value,
                         char* errstr,
                         std::size_t errstr_size) {
    if (rd_kafka_conf_set(conf, key, value.c_str(), errstr, errstr_size) != RD_KAFKA_CONF_OK) {
        throw std::runtime_error(std::string("ControlPublisher: invalid kafka conf for ") + key + ": " + errstr);
    }
}
}

ControlPublisher::ControlPublisher(const std::string& brokers, const std::string& topic) {
    char errstr[512];
    rd_kafka_conf_t* conf = rd_kafka_conf_new();
    requireKafkaConfSet(conf, "bootstrap.servers", brokers, errstr, sizeof(errstr));
    requireKafkaConfSet(conf, "linger.ms", "0", errstr, sizeof(errstr));

    producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
    if (!producer_) {
        rd_kafka_conf_destroy(conf);
        throw std::runtime_error(std::string("ControlPublisher: rd_kafka_new failed: ") + errstr);
    }
    topic_ = rd_kafka_topic_new(producer_, topic.c_str(), nullptr);
    if (!topic_) {
        rd_kafka_destroy(producer_);
        producer_ = nullptr;
        throw std::runtime_error("ControlPublisher: rd_kafka_topic_new failed");
    }
    LOG_INFO("ControlPublisher: connected to {} control_topic={}", brokers, topic);
}

ControlPublisher::~ControlPublisher() {
    if (topic_) rd_kafka_topic_destroy(topic_);
    if (producer_) {
        rd_kafka_flush(producer_, 2000);
        rd_kafka_destroy(producer_);
    }
}

std::string ControlPublisher::serializeStreamEvent(ControlEventType type,
                                                   const std::string& stream_id,
                                                   StreamState state,
                                                   double engine_ts,
                                                   int attempt,
                                                   int max_attempts,
                                                   const std::string& reason) {
    json msg = {
        {"msg_type", "stream_event"},
        {"event_type", controlEventTypeStr(type)},
        {"stream_id", stream_id},
        {"engine_ts", engine_ts},
        {"stream_state", streamStateStr(state)},
        {"attempt", attempt},
        {"max_attempts", max_attempts},
        {"reason", reason},
    };
    return msg.dump();
}

void ControlPublisher::sendRaw(const std::string& key, const std::string& payload) {
    int ret = rd_kafka_produce(
        topic_,
        RD_KAFKA_PARTITION_UA,
        RD_KAFKA_MSG_F_COPY,
        const_cast<char*>(payload.data()), payload.size(),
        key.data(), key.size(),
        nullptr);
    if (ret == -1) {
        LOG_WARN("ControlPublisher: produce failed: {}", rd_kafka_err2str(rd_kafka_last_error()));
    }
}

void ControlPublisher::publishStreamEvent(ControlEventType type,
                                          const std::string& stream_id,
                                          StreamState state,
                                          double engine_ts,
                                          int attempt,
                                          int max_attempts,
                                          const std::string& reason) {
    const std::string payload = serializeStreamEvent(type, stream_id, state, engine_ts, attempt, max_attempts, reason);
    sendRaw(stream_id, payload);
    rd_kafka_poll(producer_, 0);
}

} // namespace infer

