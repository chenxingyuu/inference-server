#pragma once

#include "common/Config.h"
#include "pipeline/IStage.h"
#include "stream/FFmpegDecoder.h"
#include <atomic>
#include <deque>
#include <mutex>
#include <string>

namespace infer {

class SourceRtspStage final : public IStage {
public:
    SourceRtspStage(std::string id, const PipelineSourceConfig& src);

    std::string id() const override;
    bool isSource() const override;

    void start() override;
    void stop() override;
    void process(const EventEnvelope&, const EmitFn& emit) override;

private:
    std::string id_;
    PipelineSourceConfig source_;
    FFmpegDecoder decoder_;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::deque<Frame> queue_;
    std::size_t max_queue_size_{64};
};

} // namespace infer
