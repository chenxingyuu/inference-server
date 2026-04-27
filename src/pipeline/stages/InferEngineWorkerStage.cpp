#include "pipeline/stages/InferEngineWorkerStage.h"

#include "common/Logger.h"
#include <algorithm>
#include <chrono>

namespace infer {

void InferEngineWorkerStage::Relay::publish(InferResult r) {
    if (owner) owner->onInferResult(std::move(r));
}

InferEngineWorkerStage::InferEngineWorkerStage(std::string id,
                                               ModelConfig model_cfg,
                                               BackendFactory backend_factory,
                                               DecoderFactory decoder_factory)
    : id_(std::move(id))
    , model_cfg_(std::move(model_cfg))
    , backend_factory_(std::move(backend_factory))
    , decoder_factory_(std::move(decoder_factory)) {
    relay_.bind(this);
    batch_deadline_ = std::chrono::steady_clock::now() +
                      std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    max_pending_ = std::max(2, model_cfg_.batch_size) * 2;

    group_ = std::make_unique<InferWorkerGroup>(
        model_cfg_,
        relay_,
        backend_factory_,
        decoder_factory_,
        nullptr,
        nullptr,
        [](const std::string&) { return TrackerType::None; },
        [](const std::string&) { return ByteTrackConfig{}; });
}

InferEngineWorkerStage::~InferEngineWorkerStage() { stop(); }

std::string InferEngineWorkerStage::id() const { return id_; }

int InferEngineWorkerStage::effectiveBatchSize() const {
    return std::max(1, model_cfg_.batch_size);
}

void InferEngineWorkerStage::start() {
    if (started_) return;
    started_ = true;
    flush_stop_.store(false);
    draining_.store(false);
    group_->start();
    LOG_INFO("InferEngineWorkerStage[{}]: model={} worker_instances={} max_queue_delay_us={}",
             id_,
             model_cfg_.id,
             group_ ? group_->instanceCount() : 0,
             model_cfg_.max_queue_delay_us);
    flush_thread_ = std::thread(&InferEngineWorkerStage::flushLoop, this);
}

void InferEngineWorkerStage::stop() {
    if (!started_) return;
    started_ = false;

    flush_stop_.store(true);
    flush_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();

    {
        std::lock_guard<std::mutex> lk(pending_mutex_);
        if (!pending_events_.empty()) {
            LOG_WARN("InferEngineWorkerStage[{}]: dropping {} pending events on stop",
                     id_,
                     pending_events_.size());
            pending_events_.clear();
        }
    }

    if (group_) group_->stop();

    {
        std::lock_guard<std::mutex> lk(inflight_mutex_);
        if (!inflight_.empty()) {
            LOG_WARN("InferEngineWorkerStage[{}]: clearing {} inflight pending results on stop",
                     id_,
                     inflight_.size());
            inflight_.clear();
        }
    }
}

void InferEngineWorkerStage::onGraphExecutorDraining() noexcept { draining_.store(true); }

std::vector<InferEngineWorkerStage::PendingEmit> InferEngineWorkerStage::extractBatch(int flush_count) {
    flush_count = std::min(flush_count, static_cast<int>(pending_events_.size()));
    std::vector<PendingEmit> out;
    out.reserve(static_cast<std::size_t>(flush_count));
    for (int i = 0; i < flush_count; ++i) {
        out.push_back(std::move(pending_events_.front()));
        pending_events_.pop_front();
    }
    batch_deadline_ = std::chrono::steady_clock::now() +
                      std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    return out;
}

void InferEngineWorkerStage::flushToWorkers(std::vector<PendingEmit> events) {
    if (events.empty()) return;

    Batch batch;
    bool have_first = false;
    {
        std::lock_guard<std::mutex> lk(inflight_mutex_);
        for (auto& pe : events) {
            const auto& ev = pe.envelope;
            if (!ev.frame) continue;
            PendingEmit stored;
            stored.emit = pe.emit;            // per-event emit, not a shared last_emit_
            stored.envelope = ev;
            stored.envelope.infer_result.reset();
            const PendingKey key{ev.stream_id, ev.frame->meta.frame_seq};
            inflight_[key] = std::move(stored);
        }
    }
    for (const auto& pe : events) {
        const auto& ev = pe.envelope;
        if (!ev.frame) continue;
        if (!have_first) {
            batch.is_gpu = ev.frame->is_gpu;
            if (batch.is_gpu) batch.cuda_device_id = ev.frame->gpu_buf.cuda_device_id;
#ifdef BUILD_ASCEND_BACKEND
            batch.is_ascend = ev.frame->is_ascend;
#endif
            have_first = true;
        }
        batch.metas.push_back(ev.frame->meta);
#ifdef BUILD_ASCEND_BACKEND
        if (batch.is_ascend) {
            batch.ascend_frames.push_back(ev.frame->ascend_buf);
        } else
#endif
        if (batch.is_gpu) {
            batch.gpu_frames.push_back(ev.frame->gpu_buf);
        } else {
            batch.frames.push_back(ev.frame->image);
        }
    }
    if (batch.empty()) return;

    std::optional<int> dev = std::nullopt;
    if (batch.is_gpu) {
        dev = batch.cuda_device_id;
    }
#ifdef BUILD_ASCEND_BACKEND
    // Ascend device affinity is resolved inside InferWorkerGroup using
    // ascend_frames[0].device_id — no extra parameter needed here.
#endif
    group_->enqueue(std::move(batch), dev);
}

void InferEngineWorkerStage::onInferResult(InferResult r) {
    EmitFn emit_copy;
    EventEnvelope out;
    {
        std::lock_guard<std::mutex> lk(inflight_mutex_);
        const PendingKey key{r.stream_id, r.frame_seq};
        auto it = inflight_.find(key);
        if (it == inflight_.end()) {
            LOG_WARN("InferEngineWorkerStage[{}]: orphan infer result stream={} seq={}",
                     id_,
                     r.stream_id,
                     r.frame_seq);
            return;
        }
        emit_copy = it->second.emit;
        out = std::move(it->second.envelope);
        inflight_.erase(it);
    }
    if (draining_.load()) return;
    out.infer_result = std::move(r);
    if (emit_copy) emit_copy(out);
}

std::size_t InferEngineWorkerStage::pendingQueueSize() {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    return pending_events_.size();
}

void InferEngineWorkerStage::flushLoop() {
    const auto half_delay = std::chrono::microseconds(
        std::max(1000, model_cfg_.max_queue_delay_us / 2));

    while (!flush_stop_.load()) {
        std::vector<PendingEmit> to_flush;
        {
            std::unique_lock<std::mutex> lk(pending_mutex_);
            flush_cv_.wait_for(lk, half_delay, [this] { return flush_stop_.load(); });
            if (flush_stop_.load()) break;

            if (!pending_events_.empty() &&
                std::chrono::steady_clock::now() >= batch_deadline_) {
                const int eff_bs = effectiveBatchSize();
                const int fc = std::min(static_cast<int>(pending_events_.size()), eff_bs);
                to_flush = extractBatch(fc);
            }
        }

        if (!to_flush.empty()) flushToWorkers(std::move(to_flush));
    }
}

void InferEngineWorkerStage::process(const EventEnvelope& input, const EmitFn& emit) {
    std::vector<PendingEmit> to_flush;
    {
        std::unique_lock<std::mutex> lk(pending_mutex_);

        if (input.frame) {
            if (static_cast<int>(pending_events_.size()) >= max_pending_) {
                LOG_WARN("InferEngineWorkerStage[{}]: pending full (size={} max={}), dropping oldest",
                         id_,
                         pending_events_.size(),
                         max_pending_);
                pending_events_.pop_front();
            }
            EventEnvelope stamped = input;
            stamped.received_at_infer_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            pending_events_.push_back({std::move(stamped), emit});  // store emit per-event
        }

        if (!pending_events_.empty()) {
            const int eff_bs = effectiveBatchSize();
            const bool hit_batch = static_cast<int>(pending_events_.size()) >= eff_bs;
            const bool hit_deadline = std::chrono::steady_clock::now() >= batch_deadline_;
            if (hit_batch || hit_deadline) {
                const int fc = std::min(static_cast<int>(pending_events_.size()), eff_bs);
                to_flush = extractBatch(fc);
            }
        }
    }

    if (!to_flush.empty()) flushToWorkers(std::move(to_flush));
}

} // namespace infer
