#pragma once

#include "common/Config.h"
#include "stream/StreamHealthRegistry.h"
#include <string>
#include <thread>
#include <atomic>

// Forward-declare librdkafka handle to avoid leaking C headers into callers
struct rd_kafka_s;
struct rd_kafka_topic_s;

namespace infer {

// HeartbeatPublisher emits per-stream and engine-level heartbeat messages to a
// dedicated Kafka topic at a fixed interval.  Runs on its own background thread.
//
// Purpose: lets downstream consumers distinguish three silence conditions:
//   1) Engine crashed          → heartbeats stop entirely
//   2) Camera disconnected     → stream_state = RECONNECTING or DEGRADED
//   3) No detections this interval → stream_state = STREAMING, frames_since_last_hb = 0
class HeartbeatPublisher {
public:
    explicit HeartbeatPublisher(const KafkaConfig& cfg);
    ~HeartbeatPublisher();

    HeartbeatPublisher(const HeartbeatPublisher&)            = delete;
    HeartbeatPublisher& operator=(const HeartbeatPublisher&) = delete;

    void start();
    void stop();

    // Exposed for unit testing: serialise a single per-stream heartbeat JSON.
    static std::string serializeStreamHb(const std::string& stream_id,
                                         const StreamHealth& h,
                                         double engine_ts);

    // Exposed for unit testing: serialise the engine-level heartbeat JSON.
    static std::string serializeEngineHb(int active_streams, double engine_ts);

private:
    void loop();
    void sendRaw(const std::string& key, const std::string& payload);
    void updateMetrics(const std::vector<std::pair<std::string, StreamHealth>>& all,
                       double now_ts);

    rd_kafka_s*       producer_{nullptr};
    rd_kafka_topic_s* topic_{nullptr};
    int               interval_ms_;
    std::thread       thread_;
    std::atomic<bool> stop_{false};
};

} // namespace infer
