#pragma once

#include "common/Config.h"
#include "publisher/IPublisher.h"
#include "archive/FrameArchiver.h"
#include "infer/BackendFactory.h"
#include "decoder/DecoderFactory.h"
#include "pipeline/IStage.h"
#include <memory>
#include <string>

namespace infer {

class StageFactory {
public:
    struct Context {
        const AppConfig& app_config;
        const PipelineSourceConfig& source;
        IPublisher& publisher;
        std::shared_ptr<FrameArchiver> frame_archiver;
    };

    static std::unique_ptr<IStage> create(const StageConfig& cfg, const Context& ctx);
};

} // namespace infer
