#pragma once

#include "pipeline/IStage.h"
#include <string>

namespace infer {

// Placeholder stages (`decode.ffmpeg`, `preprocess.yolo`, `postprocess.yolo`) forward events unchanged.
class PassthroughStage final : public IStage {
public:
    explicit PassthroughStage(std::string id) : id_(std::move(id)) {}
    std::string id() const override { return id_; }
    void process(const EventEnvelope& input, const EmitFn& emit) override { emit(input); }

private:
    std::string id_;
};

} // namespace infer
