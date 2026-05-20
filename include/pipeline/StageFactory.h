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
#include <unordered_map>

namespace infer {

class StageFactory {
public:
    struct Context {
        const AppConfig& app_config;
        const PipelineSourceConfig& source;
        // Named publisher registry. sink.publish reads with.to to pick targets;
        // if with.to is absent, all publishers are used (fan-out).
        const std::unordered_map<std::string, IPublisher*>& publishers;
        std::shared_ptr<FrameArchiver> frame_archiver;
        int          ingest_sample_fps{5};
        SamplingMode ingest_sampling_mode{SamplingMode::FrameCount};
        bool         ingest_use_hwdec{false};
        bool         ingest_use_ascend_dvpp{false};
        int          ingest_ascend_device_id{0};
        int          ingest_vpc_out_width{0};
        int          ingest_vpc_out_height{0};
        std::shared_ptr<GpuDeviceAllocator> gpu_alloc; // null if hwdec disabled
    };

    static std::unique_ptr<IStage> create(const StageConfig& cfg, const Context& ctx);
};

} // namespace infer
