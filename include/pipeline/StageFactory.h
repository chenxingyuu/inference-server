#pragma once

#include "common/Config.h"
#include "publisher/IPublisher.h"
#include "archive/FrameArchiver.h"
#include "infer/BackendFactory.h"
#include "decoder/DecoderFactory.h"
#include "pipeline/IStage.h"
#include "stream/GpuDeviceAllocator.h"
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
        int          ingest_sample_fps{5};
        SamplingMode ingest_sampling_mode{SamplingMode::FrameCount};
        bool         ingest_use_hwdec{false};
        std::shared_ptr<GpuDeviceAllocator> gpu_alloc; // null if hwdec disabled
    };

    static std::unique_ptr<IStage> create(const StageConfig& cfg, const Context& ctx);
};

} // namespace infer
