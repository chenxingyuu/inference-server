#include "pipeline/stages/SinkKafkaStage.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"

#include <chrono>

namespace infer {

namespace {

uint64_t nowSteadyNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

double nsToMs(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

} // namespace

SinkKafkaStage::SinkKafkaStage(std::string id, IPublisher& publisher)
    : id_(std::move(id)), pub_(&publisher) {}

SinkKafkaStage::SinkKafkaStage(std::string id, std::unique_ptr<IPublisher> publisher)
    : id_(std::move(id)), owned_(std::move(publisher)), pub_(owned_.get()) {}

std::string SinkKafkaStage::id() const { return id_; }

void SinkKafkaStage::process(const EventEnvelope& input, const EmitFn&) {
    if (!input.infer_result) return;
    if (input.infer_result->frame_mono_ns != 0) {
        const uint64_t now_ns = nowSteadyNs();
        if (now_ns >= input.infer_result->frame_mono_ns) {
            Metrics::get().recordSinkPublishLatency(
                input.infer_result->stream_id,
                nsToMs(now_ns - input.infer_result->frame_mono_ns));
        }
    }
    LOG_DEBUG("SinkKafkaStage[{}]: publish stream={} frame_seq={} dets={} latency_ms={:.1f} queue_latency_ms={:.1f}",
              id_,
              input.infer_result->stream_id,
              input.infer_result->frame_seq,
              input.infer_result->detections.size(),
              input.infer_result->latency_ms,
              input.infer_result->queue_latency_ms);
    pub_->publish(*input.infer_result);
}

} // namespace infer
