#include "pipeline/stages/InferEngineStage.h"

#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <algorithm>
#include <chrono>

namespace infer {

namespace {

double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

uint64_t nowSteadyNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

double nsToMs(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

} // namespace

InferEngineStage::InferEngineStage(std::string id,
                                 ModelConfig model_cfg,
                                 std::unique_ptr<IInferBackend> backend,
                                 std::unique_ptr<IYOLODecoder> decoder)
    : id_(std::move(id))
    , model_cfg_(std::move(model_cfg))
    , backend_(std::move(backend))
    , decoder_(std::move(decoder)) {
    batch_deadline_ = std::chrono::steady_clock::now() + std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    max_pending_ = std::max(2, model_cfg_.batch_size) * 2;
}

std::string InferEngineStage::id() const { return id_; }

void InferEngineStage::start() {
    // Load lazily on first inference.
}

void InferEngineStage::stop() {
    if (loaded_) {
        backend_->unloadModel();
        loaded_ = false;
    }
}

void InferEngineStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (input.frame) {
        if (static_cast<int>(pending_events_.size()) >= max_pending_) {
            LOG_WARN("InferEngineStage[{}]: pending queue full ({} frames), dropping oldest", id_, max_pending_);
            pending_events_.erase(pending_events_.begin());
        }
        pending_events_.push_back(input);
    }
    const auto now = std::chrono::steady_clock::now();
    if (pending_events_.empty()) return;
    const int effective_batch_size = fallback_to_single_infer_
        ? 1
        : std::max(1, std::min(model_cfg_.batch_size, backend_->maxBatchSize()));
    const bool hit_batch = static_cast<int>(pending_events_.size()) >= effective_batch_size;
    const bool hit_deadline = now >= batch_deadline_;
    if (!hit_batch && !hit_deadline) return;

    const int flush_count = std::min(static_cast<int>(pending_events_.size()), effective_batch_size);
    try {
        // 延迟加载模型：仅在首次需要推理时执行一次。
        if (!loaded_) {
            backend_->loadModel(model_cfg_);
            loaded_ = true;
        }
        Batch batch;
        batch.is_gpu = pending_events_.front().frame->is_gpu;
        for (int i = 0; i < flush_count; ++i) {
            const auto& ev = pending_events_[i];
            batch.metas.push_back(ev.frame->meta);
            if (batch.is_gpu) batch.gpu_frames.push_back(ev.frame->gpu_buf);
            else batch.frames.push_back(ev.frame->image);
        }
        std::vector<float> output;
        const auto infer_start = std::chrono::steady_clock::now();
        backend_->infer(batch, output);
        const auto infer_end_tp = std::chrono::steady_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end_tp - infer_start).count();

        Metrics::get().recordInferLatency(model_cfg_.id, infer_ms);
        Metrics::get().recordInferBatchSize(model_cfg_.id, batch.size());
        Metrics::get().incInferBatches(model_cfg_.id);

        const auto decode_start = std::chrono::steady_clock::now();
        auto decoded = decoder_->decode(output.data(), batch.size(), model_cfg_.input_shape,
                                        model_cfg_.conf_thresh, model_cfg_.nms_thresh);
        const double decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_start).count();

        const uint64_t infer_finish_mono_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const double infer_ts = nowEpoch();
        for (int i = 0; i < batch.size(); ++i) {
            EventEnvelope out = pending_events_[i];
            InferResult result;
            result.stream_id = out.stream_id;
            result.frame_seq = out.frame_seq;
            result.frame_ts = out.frame->meta.capture_ts;
            result.frame_mono_ns = out.frame->meta.capture_mono_ns;
            result.infer_ts = infer_ts;
            result.infer_ms = infer_ms;
            result.decode_ms = decode_ms;
            if (result.frame_mono_ns != 0) {
                const uint64_t batch_start_mono_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        infer_start.time_since_epoch()).count());
                result.queue_latency_ms = nsToMs(batch_start_mono_ns - result.frame_mono_ns);
                result.latency_ms = nsToMs(infer_finish_mono_ns - result.frame_mono_ns);
            } else {
                // Fallback: epoch-based duration (may be affected by system clock jumps).
                result.latency_ms = (infer_ts - result.frame_ts) * 1000.0;
                result.queue_latency_ms = 0.0;
            }
            result.model_id = model_cfg_.id;
            if (i < static_cast<int>(decoded.size())) {
                result.detections = std::move(decoded[i]);
            }
            out.infer_result = std::move(result);
            emit(out);
        }
    } catch (const std::exception& e) {
        const std::string err = e.what();
        if (err.find("invalid dimensions") != std::string::npos && flush_count > 1) {
            LOG_WARN("InferEngineStage[{}]: batch inference failed with dimension mismatch, fallback to bs=1",
                     id_);
            fallback_to_single_infer_ = true;
            for (int i = 0; i < flush_count; ++i) {
                try {
                    Batch one;
                    one.is_gpu = pending_events_[i].frame->is_gpu;
                    one.metas.push_back(pending_events_[i].frame->meta);
                    if (one.is_gpu) one.gpu_frames.push_back(pending_events_[i].frame->gpu_buf);
                    else one.frames.push_back(pending_events_[i].frame->image);

                    std::vector<float> output;
                    const auto infer_start = std::chrono::steady_clock::now();
                    backend_->infer(one, output);
                    const auto infer_end_tp = std::chrono::steady_clock::now();
                    const double infer_ms = std::chrono::duration<double, std::milli>(infer_end_tp - infer_start).count();

                    Metrics::get().recordInferLatency(model_cfg_.id, infer_ms);
                    Metrics::get().recordInferBatchSize(model_cfg_.id, 1);
                    Metrics::get().incInferBatches(model_cfg_.id);

                    const auto decode_start = std::chrono::steady_clock::now();
                    auto decoded = decoder_->decode(output.data(), 1, model_cfg_.input_shape,
                                                    model_cfg_.conf_thresh, model_cfg_.nms_thresh);
                    const double decode_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - decode_start).count();

                    EventEnvelope out = pending_events_[i];
                    InferResult result;
                    result.stream_id = out.stream_id;
                    result.frame_seq = out.frame_seq;
                    result.frame_ts = out.frame->meta.capture_ts;
                    result.frame_mono_ns = out.frame->meta.capture_mono_ns;
                    result.infer_ts = nowEpoch();
                    result.infer_ms = infer_ms;
                    result.decode_ms = decode_ms;
                    if (result.frame_mono_ns != 0) {
                        const uint64_t batch_start_mono_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                infer_start.time_since_epoch()).count());
                        const uint64_t infer_finish_mono_ns = nowSteadyNs();
                        result.queue_latency_ms = nsToMs(batch_start_mono_ns - result.frame_mono_ns);
                        result.latency_ms = nsToMs(infer_finish_mono_ns - result.frame_mono_ns);
                    } else {
                        result.latency_ms = (result.infer_ts - result.frame_ts) * 1000.0;
                        result.queue_latency_ms = 0.0;
                    }
                    result.model_id = model_cfg_.id;
                    if (!decoded.empty()) result.detections = std::move(decoded[0]);
                    out.infer_result = std::move(result);
                    emit(out);
                } catch (const std::exception& one_e) {
                    LOG_ERROR("InferEngineStage[{}]: fallback bs=1 failed: {}", id_, one_e.what());
                }
            }
        } else {
            LOG_ERROR("InferEngineStage[{}]: inference failed: {}", id_, err);
            // Emit envelopes without infer_result so downstream stages are not starved.
            for (int i = 0; i < flush_count; ++i) {
                emit(pending_events_[i]);
            }
        }
    }
    pending_events_.erase(pending_events_.begin(), pending_events_.begin() + flush_count);
    batch_deadline_ = std::chrono::steady_clock::now() + std::chrono::microseconds(model_cfg_.max_queue_delay_us);
}

} // namespace infer
