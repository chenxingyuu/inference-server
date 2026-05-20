#pragma once

#include "pipeline/IStage.h"
#include "publisher/IPublisher.h"
#include <memory>
#include <string>

namespace infer {

class SinkKafkaStage final : public IStage {
public:
    // Non-owning: publisher must outlive this stage.
    SinkKafkaStage(std::string id, IPublisher& publisher);

    // Owning: stage takes exclusive ownership (used for ad-hoc MultiPublisher).
    SinkKafkaStage(std::string id, std::unique_ptr<IPublisher> publisher);

    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn&) override;

private:
    std::string                  id_;
    std::unique_ptr<IPublisher>  owned_;   // non-null only for owning variant
    IPublisher*                  pub_;     // always valid
};

} // namespace infer
