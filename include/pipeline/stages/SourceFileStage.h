#pragma once

#include "common/Config.h"
#include "pipeline/IStage.h"
#include "stream/FFmpegDecoder.h"
#include <atomic>
#include <deque>
#include <mutex>
#include <string>

namespace infer {

// Pipeline source stage that reads from a local video file via FFmpeg.
// Unlike SourceRtspStage, this stage treats EOF as a clean terminal condition
// and does not attempt to reconnect. Set loop=true to repeat the file.
class SourceFileStage final : public IStage {
public:
    SourceFileStage(std::string id, const PipelineSourceConfig& src,
                    int sample_fps, bool use_hwdec, bool loop);

    std::string id() const override;
    bool isSource() const override;

    void start() override;
    void stop() override;
    void process(const EventEnvelope&, const EmitFn& emit) override;

private:
    std::string id_;
    PipelineSourceConfig source_;
    int sample_fps_;
    bool use_hwdec_;
    bool loop_;
    FFmpegDecoder decoder_;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::deque<Frame> queue_;
    std::size_t max_queue_size_{64};
};

} // namespace infer
