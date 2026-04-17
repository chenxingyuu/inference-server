#pragma once

#include "common/Config.h"
#include "decoder/IYOLODecoder.h"
#include "infer/IInferBackend.h"
#include "pipeline/IStage.h"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace infer {

class InferEngineStage final : public IStage {
public:
    InferEngineStage(std::string id,
                     ModelConfig model_cfg,
                     std::unique_ptr<IInferBackend> backend,
                     std::unique_ptr<IYOLODecoder> decoder);

    std::string id() const override;

    void start() override;
    void stop() override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    std::string id_;
    ModelConfig model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder> decoder_;
    std::vector<EventEnvelope> pending_events_;
    std::chrono::steady_clock::time_point batch_deadline_;
    bool fallback_to_single_infer_{false};
    bool loaded_{false};
};

} // namespace infer
