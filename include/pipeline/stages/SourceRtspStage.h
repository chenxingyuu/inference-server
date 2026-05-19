#pragma once

#include "common/Config.h"
#include "pipeline/IStage.h"
#include "stream/FFmpegDecoder.h"
#include "stream/GpuDeviceAllocator.h"
#ifdef BUILD_ASCEND_BACKEND
#include "stream/DVPPDecoder.h"
#endif
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

namespace infer {

class SourceRtspStage final : public IStage {
public:
    SourceRtspStage(std::string id, const PipelineSourceConfig& src,
                    int sample_fps, SamplingMode sampling_mode, bool use_hwdec,
                    std::shared_ptr<GpuDeviceAllocator> gpu_alloc = nullptr,
                    bool use_ascend_dvpp = false, int ascend_device_id = 0,
                    int vpc_out_width = 0, int vpc_out_height = 0);

    std::string id() const override;
    bool isSource() const override;

    void start() override;
    void stop() override;
    void process(const EventEnvelope&, const EmitFn& emit) override;

private:
    std::string id_;
    PipelineSourceConfig source_;
    int sample_fps_;
    SamplingMode sampling_mode_;
    bool use_hwdec_;
    std::shared_ptr<GpuDeviceAllocator> gpu_alloc_;
    int cuda_device_id_{0};
    FFmpegDecoder decoder_;
#ifdef BUILD_ASCEND_BACKEND
    std::unique_ptr<DVPPDecoder> dvpp_decoder_;
    bool use_ascend_dvpp_{false};
    int  ascend_device_id_{0};
    int  vpc_out_width_{0};
    int  vpc_out_height_{0};
#endif
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Frame> queue_;
    std::size_t max_queue_size_{64};
};

} // namespace infer
