#include "pipeline/stages/SourceRtspStage.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace infer {

SourceRtspStage::SourceRtspStage(std::string id, const PipelineSourceConfig& src, int sample_fps, bool use_hwdec)
    : id_(std::move(id))
    , source_(src)
    , sample_fps_(sample_fps)
    , use_hwdec_(use_hwdec)
    , max_queue_size_(std::max(std::size_t{32}, static_cast<std::size_t>(sample_fps * 2))) {}

std::string SourceRtspStage::id() const { return id_; }

bool SourceRtspStage::isSource() const { return true; }

void SourceRtspStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    StreamConfig cfg;
    cfg.id = source_.id;
    cfg.url = source_.url;
    cfg.sample_fps = sample_fps_;
    cfg.reconnect_delay_ms = source_.reconnect_delay_ms;
    cfg.max_reconnect_delay_ms = source_.max_reconnect_delay_ms;
    cfg.degraded_threshold = source_.degraded_threshold;
    cfg.max_reconnect_attempts = source_.max_reconnect_attempts;
    cfg.use_hwdec = use_hwdec_;
    decoder_.start(cfg, [this](Frame frame) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.size() >= max_queue_size_) {
            queue_.pop_front();
        }
        queue_.push_back(std::move(frame));
    });
}

void SourceRtspStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    decoder_.stop();
}

void SourceRtspStage::process(const EventEnvelope&, const EmitFn& emit) {
    Frame frame;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            return;
        }
        frame = std::move(queue_.front());
        queue_.pop_front();
    }
    EventEnvelope out;
    out.event_id = frame.meta.stream_id + ":" + std::to_string(frame.meta.frame_seq);
    out.stream_id = frame.meta.stream_id;
    out.frame_seq = frame.meta.frame_seq;
    out.frame = std::make_shared<Frame>(std::move(frame));
    emit(out);
}

} // namespace infer
