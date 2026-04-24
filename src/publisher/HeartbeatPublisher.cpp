#include "publisher/HeartbeatPublisher.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"
#include <nlohmann/json.hpp>
#include <librdkafka/rdkafka.h>
#include <chrono>
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
        throw std::runtime_error(std::string("HeartbeatPublisher: invalid kafka conf for ") + key + ": " + errstr);
    }
}
double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}
} // namespace

HeartbeatPublisher::HeartbeatPublisher(const KafkaConfig& cfg)
    : interval_ms_(cfg.heartbeat_interval_ms)
{
    char errstr[512];
    rd_kafka_conf_t* conf = rd_kafka_conf_new();
    requireKafkaConfSet(conf, "bootstrap.servers", cfg.brokers, errstr, sizeof(errstr));
    // heartbeat messages are tiny — no batching/compression needed
    requireKafkaConfSet(conf, "linger.ms", "0", errstr, sizeof(errstr));

    producer_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
    if (!producer_) {
        rd_kafka_conf_destroy(conf);
        throw std::runtime_error(
            std::string("HeartbeatPublisher: rd_kafka_new failed: ") + errstr);
    }

    topic_ = rd_kafka_topic_new(producer_, cfg.heartbeat_topic.c_str(), nullptr);
    if (!topic_) {
        rd_kafka_destroy(producer_);
        producer_ = nullptr;
        throw std::runtime_error("HeartbeatPublisher: rd_kafka_topic_new failed");
    }

    LOG_INFO("HeartbeatPublisher: connected to {} heartbeat_topic={} interval={}ms",
             cfg.brokers, cfg.heartbeat_topic, interval_ms_);
}

HeartbeatPublisher::~HeartbeatPublisher() {
    stop();
    if (topic_)    rd_kafka_topic_destroy(topic_);
    if (producer_) {
        rd_kafka_flush(producer_, 2000);
        rd_kafka_destroy(producer_);
    }
}

void HeartbeatPublisher::start() {
    stop_.store(false);
    thread_ = std::thread(&HeartbeatPublisher::loop, this);
}

void HeartbeatPublisher::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

void HeartbeatPublisher::loop() {
    LOG_INFO("HeartbeatPublisher: thread started (interval={}ms)", interval_ms_);
    auto& reg = StreamHealthRegistry::get();

    while (!stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        if (stop_.load()) break;

        double ts = nowEpoch();
        auto all = reg.getAllHealth();

        // Per-stream heartbeats
        for (const auto& [stream_id, h] : all) {
            std::string payload = serializeStreamHb(stream_id, h, ts);
            sendRaw(stream_id, payload);
        }

        // Engine-level heartbeat (key = "__engine__")
        sendRaw("__engine__", serializeEngineHb(static_cast<int>(all.size()), ts));

        // Update Prometheus metrics and reset per-hb frame counter
        updateMetrics(all, ts);
        reg.resetHbCounters();

        rd_kafka_poll(producer_, 0);
    }
    LOG_INFO("HeartbeatPublisher: thread stopped");
}

void HeartbeatPublisher::sendRaw(const std::string& key, const std::string& payload) {
    int ret = rd_kafka_produce(
        topic_,
        RD_KAFKA_PARTITION_UA,
        RD_KAFKA_MSG_F_COPY,
        const_cast<char*>(payload.data()), payload.size(),
        key.data(), key.size(),
        nullptr);
    if (ret == -1) {
        LOG_WARN("HeartbeatPublisher: produce failed: {}",
                 rd_kafka_err2str(rd_kafka_last_error()));
    }
}

void HeartbeatPublisher::updateMetrics(
    const std::vector<std::pair<std::string, StreamHealth>>& all,
    double now_ts)
{
    auto& m = Metrics::get();
    for (const auto& [stream_id, h] : all) {
        m.setStreamState(stream_id, static_cast<uint32_t>(h.state));
        m.setStreamReconnectCount(stream_id, h.reconnect_count);
        m.setStreamConsecutiveFailures(stream_id, h.consecutive_failures);
        double age = (h.last_frame_ts > 0.0) ? (now_ts - h.last_frame_ts) : -1.0;
        m.setStreamLastFrameAgeSeconds(stream_id, age);
    }
}

// ── Static serialisation (testable without Kafka) ─────────────────────────────

std::string HeartbeatPublisher::serializeStreamHb(const std::string& stream_id,
                                                    const StreamHealth& h,
                                                    double engine_ts)
{
    json msg = {
        {"msg_type",              "heartbeat"},
        {"stream_id",             stream_id},
        {"engine_ts",             engine_ts},
        {"stream_state",          streamStateStr(h.state)},
        {"reconnect_count",       h.reconnect_count},
        {"consecutive_failures",  h.consecutive_failures},
        {"frames_since_last_hb",  h.frames_since_last_hb},
        {"engine_healthy",        true}
    };
    return msg.dump();
}

std::string HeartbeatPublisher::serializeEngineHb(int active_streams, double engine_ts) {
    json msg = {
        {"msg_type",       "engine_heartbeat"},
        {"stream_id",      nullptr},
        {"engine_ts",      engine_ts},
        {"active_streams", active_streams},
        {"engine_healthy", true}
    };
    return msg.dump();
}

} // namespace infer
