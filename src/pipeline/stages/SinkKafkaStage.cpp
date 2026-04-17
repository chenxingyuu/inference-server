#include "pipeline/stages/SinkKafkaStage.h"

namespace infer {

SinkKafkaStage::SinkKafkaStage(std::string id, IPublisher& publisher)
    : id_(std::move(id)), publisher_(publisher) {}

std::string SinkKafkaStage::id() const { return id_; }

void SinkKafkaStage::process(const EventEnvelope& input, const EmitFn&) {
    if (input.infer_result) publisher_.publish(*input.infer_result);
}

} // namespace infer
