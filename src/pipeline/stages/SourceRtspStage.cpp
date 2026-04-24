#include "pipeline/stages/SourceRtspStage.h"

#include "metrics/Metrics.h"

#include <algorithm>
#include <chrono>

namespace infer {

SourceRtspStage::SourceRtspStage(std::string id, const PipelineSourceConfig& src,
                                 int sample_fps, SamplingMode sampling_mode, bool use_hwdec)
    : id_(std::move(id))
    , source_(src)
    , sample_fps_(sample_fps)
    , sampling_mode_(sampling_mode)
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
    cfg.sample_fps    = sample_fps_;
    cfg.sampling_mode = sampling_mode_;
    cfg.reconnect_delay_ms = source_.reconnect_delay_ms;
    cfg.max_reconnect_delay_ms = source_.max_reconnect_delay_ms;
    cfg.degraded_threshold = source_.degraded_threshold;
    cfg.max_reconnect_attempts = source_.max_reconnect_attempts;
    cfg.open_timeout_ms = source_.open_timeout_ms;
    cfg.read_timeout_ms = source_.read_timeout_ms;
    cfg.use_hwdec = use_hwdec_;
    decoder_.start(cfg, [this](Frame frame) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.size() >= max_queue_size_) {
            queue_.pop_front();
            Metrics::get().incFramesDropped(source_.id);
        }
        queue_.push_back(std::move(frame));
        cv_.notify_one();
    });
}

void SourceRtspStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    cv_.notify_all();
    decoder_.stop();
}

void SourceRtspStage::process(const EventEnvelope&, const EmitFn& emit) {
    Frame frame;
    std::size_t source_queue_size = 0;
    {
        std::unique_lock<std::mutex> lock(mu_);
        if (queue_.empty()) {
            cv_.wait_for(lock, std::chrono::milliseconds(2), [this] { return !queue_.empty() || !running_.load(); });
        }
        if (queue_.empty()) {
            return;
        }
        frame = std::move(queue_.front());
        queue_.pop_front();
        source_queue_size = queue_.size();
    }
    EventEnvelope out;
    out.event_id = frame.meta.stream_id + ":" + std::to_string(frame.meta.frame_seq);
    out.stream_id = frame.meta.stream_id;
    out.frame_seq = frame.meta.frame_seq;
    out.source_queue_size = source_queue_size;
    out.frame = std::make_shared<Frame>(std::move(frame));
    emit(out);
}

} // namespace infer
