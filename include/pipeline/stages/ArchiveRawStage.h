#pragma once

#include "archive/FrameArchiver.h"
#include "pipeline/IStage.h"
#include <memory>
#include <string>

namespace infer {

class ArchiveRawStage final : public IStage {
public:
    ArchiveRawStage(std::string id, std::shared_ptr<FrameArchiver> archiver, bool allow_gpu_frames);

    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    std::string id_;
    std::shared_ptr<FrameArchiver> archiver_;
    bool allow_gpu_frames_{true};
};

} // namespace infer
