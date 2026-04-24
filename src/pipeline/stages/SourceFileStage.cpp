#include "pipeline/stages/SourceFileStage.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace infer {

SourceFileStage::SourceFileStage(std::string id, const PipelineSourceConfig& src,
                                 int sample_fps, SamplingMode sampling_mode, bool use_hwdec, bool loop)
    : id_(std::move(id))
    , source_(src)
    , sample_fps_(sample_fps)
    , sampling_mode_(sampling_mode)
    , use_hwdec_(use_hwdec)
    , loop_(loop)
    , max_queue_size_(std::max(std::size_t{32}, static_cast<std::size_t>(sample_fps * 2))) {}

std::string SourceFileStage::id() const { return id_; }

bool SourceFileStage::isSource() const { return true; }

void SourceFileStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    StreamConfig cfg;
    cfg.id                   = source_.id;
    cfg.url                  = source_.url;
    cfg.sample_fps    = sample_fps_;
    cfg.sampling_mode = sampling_mode_;
    cfg.open_timeout_ms      = source_.open_timeout_ms;
    cfg.read_timeout_ms      = source_.read_timeout_ms;
    cfg.use_hwdec            = use_hwdec_;
    cfg.source_type          = SourceType::File;
    cfg.loop_file            = loop_;
    cfg.max_reconnect_attempts = 0;  // never reconnect for file sources
    decoder_.start(cfg, [this](Frame frame) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.size() >= max_queue_size_) {
            queue_.pop_front();
        }
        queue_.push_back(std::move(frame));
    });
}

void SourceFileStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    decoder_.stop();
}

void SourceFileStage::process(const EventEnvelope&, const EmitFn& emit) {
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
    out.event_id  = frame.meta.stream_id + ":" + std::to_string(frame.meta.frame_seq);
    out.stream_id = frame.meta.stream_id;
    out.frame_seq = frame.meta.frame_seq;
    out.frame     = std::make_shared<Frame>(std::move(frame));
    emit(out);
}

} // namespace infer
