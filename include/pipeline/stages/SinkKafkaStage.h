#pragma once

#include "pipeline/IStage.h"
#include "publisher/IPublisher.h"
#include <string>

namespace infer {

class SinkKafkaStage final : public IStage {
public:
    SinkKafkaStage(std::string id, IPublisher& publisher);

    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn&) override;

private:
    std::string id_;
    IPublisher& publisher_;
};

} // namespace infer
