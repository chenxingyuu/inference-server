#pragma once

#include "stream/StreamHealthRegistry.h"
#include <atomic>
#include <string>
#include <thread>

// Forward-declare librdkafka handle to avoid leaking C headers into callers
struct rd_kafka_s;
struct rd_kafka_topic_s;

namespace infer {

enum class ControlEventType {
    StreamDropped,
    StreamRecovered,
    StreamFailedTerminal,
};

inline const char* controlEventTypeStr(ControlEventType t) {
    switch (t) {
        case ControlEventType::StreamDropped:        return "stream_dropped";
        case ControlEventType::StreamRecovered:      return "stream_recovered";
        case ControlEventType::StreamFailedTerminal: return "stream_failed_terminal";
    }
    return "unknown";
}

// ControlPublisher emits stream-level control events to a dedicated Kafka topic.
class ControlPublisher {
public:
    ControlPublisher(const std::string& brokers, const std::string& topic);
    ~ControlPublisher();

    ControlPublisher(const ControlPublisher&)            = delete;
    ControlPublisher& operator=(const ControlPublisher&) = delete;

    void publishStreamEvent(ControlEventType type,
                            const std::string& stream_id,
                            StreamState state,
                            double engine_ts,
                            int attempt,
                            int max_attempts,
                            const std::string& reason);

    // Exposed for unit testing: serialise a stream control event JSON.
    static std::string serializeStreamEvent(ControlEventType type,
                                            const std::string& stream_id,
                                            StreamState state,
                                            double engine_ts,
                                            int attempt,
                                            int max_attempts,
                                            const std::string& reason);

private:
    void sendRaw(const std::string& key, const std::string& payload);

    rd_kafka_s*       producer_{nullptr};
    rd_kafka_topic_s* topic_{nullptr};
};

} // namespace infer

