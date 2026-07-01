#include "pipeline/stages/InferEngineStage.h"

#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "pipeline/stages/DetectionOverlay.h"
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
    batch_deadline_ = std::chrono::steady_clock::now() +
                      std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    max_pending_ = std::max(2, model_cfg_.batch_size) * 2;
}

std::string InferEngineStage::id() const { return id_; }

void InferEngineStage::start() {
    flush_stop_.store(false);
    flush_thread_ = std::thread(&InferEngineStage::flushLoop, this);
}

void InferEngineStage::stop() {
    flush_stop_.store(true);
    flush_cv_.notify_all();
    if (flush_thread_.joinable()) flush_thread_.join();
    if (loaded_) {
        backend_->unloadModel();
        loaded_ = false;
    }
}

// Called with pending_mutex_ held. Pops frames and resets deadline; returns the batch.
std::vector<EventEnvelope> InferEngineStage::extractBatch(int flush_count) {
    flush_count = std::min(flush_count, static_cast<int>(pending_events_.size()));
    std::vector<EventEnvelope> out;
    out.reserve(flush_count);
    for (int i = 0; i < flush_count; ++i) {
        out.push_back(std::move(pending_events_.front()));
        pending_events_.pop_front();
    }
    batch_deadline_ = std::chrono::steady_clock::now() +
                      std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    return out;
}

// Runs TRT inference + decode + emit. Called WITHOUT pending_mutex_ held.
// Serialized by infer_mutex_ so GPU is never called concurrently from two threads.
void InferEngineStage::inferAndEmit(std::vector<EventEnvelope> events, const EmitFn& emit) {
    if (events.empty()) return;

    const int flush_count = static_cast<int>(events.size());
    std::vector<EventEnvelope> outputs;
    outputs.reserve(flush_count);

    try {
        Batch batch;
        std::vector<float> output;
        double infer_ms = 0.0;
        double decode_ms = 0.0;
        uint64_t batch_start_ns = 0;
        uint64_t finish_ns = 0;
        double infer_ts = 0.0;
        std::vector<std::vector<Detection>> decoded;
        {
            std::lock_guard infer_lock(infer_mutex_);
            if (!loaded_) {
                backend_->loadModel(model_cfg_);
                loaded_ = true;
            }

            batch.is_gpu = events.front().frame->is_gpu;
            for (const auto& ev : events) {
                batch.metas.push_back(ev.frame->meta);
                if (batch.is_gpu) batch.gpu_frames.push_back(ev.frame->gpu_buf);
                else batch.frames.push_back(ev.frame->image);
            }

            const auto infer_start = std::chrono::steady_clock::now();
            backend_->infer(batch, output);
            const auto infer_end_tp = std::chrono::steady_clock::now();
            infer_ms = std::chrono::duration<double, std::milli>(
                infer_end_tp - infer_start).count();

            const auto decode_start = std::chrono::steady_clock::now();
            decoded = decoder_->decode(output.data(), batch.size(), model_cfg_.input_shape,
                                       model_cfg_.conf_thresh, model_cfg_.nms_thresh,
                                       output.size());
            decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();

            batch_start_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    infer_start.time_since_epoch()).count());
            finish_ns = nowSteadyNs();
            infer_ts = nowEpoch();
        }

        Metrics::get().recordInferLatency(model_cfg_.id, infer_ms);
        Metrics::get().recordInferBatchSize(model_cfg_.id, batch.size());
        Metrics::get().incInferBatches(model_cfg_.id);
        for (int i = 0; i < flush_count; ++i) {
            const uint64_t cap_ns = events[i].frame->meta.capture_mono_ns;
            if (cap_ns != 0 && batch_start_ns >= cap_ns)
                Metrics::get().recordInferQueueLatency(model_cfg_.id, nsToMs(batch_start_ns - cap_ns));
        }

        const std::size_t pending_q = pendingQueueSize();
        // Log per-frame queue latency so we can see how long frames waited before inference.
        for (int i = 0; i < flush_count; ++i) {
            const uint64_t cap_ns = events[i].frame->meta.capture_mono_ns;
            if (cap_ns != 0 && batch_start_ns >= cap_ns) {
                const double q_ms = nsToMs(batch_start_ns - cap_ns);
                const uint64_t recv_ns = events[i].received_at_infer_ns;
                const double transit_ms = (recv_ns != 0 && recv_ns >= cap_ns)
                    ? nsToMs(recv_ns - cap_ns) : -1.0;
                const double pending_ms = (recv_ns != 0 && batch_start_ns >= recv_ns)
                    ? nsToMs(batch_start_ns - recv_ns) : -1.0;
                const std::size_t source_q = events[i].source_queue_size.value_or(0);
                const std::string ingress_edge = events[i].ingress_edge.value_or("unknown");
                const std::size_t ingress_edge_q = events[i].ingress_edge_queue_size.value_or(0);
                LOG_DEBUG("InferEngineStage[{}]: seq={} queue_latency_ms={:.1f} transit_ms={:.1f} pending_ms={:.1f} infer_ms={:.1f} pending_queue={} source_queue={} ingress_edge={} ingress_edge_queue={}",
                         id_,
                         events[i].frame->meta.frame_seq,
                         q_ms,
                         transit_ms,
                         pending_ms,
                         infer_ms,
                         pending_q,
                         source_q,
                         ingress_edge,
                         ingress_edge_q);
            }
        }
        for (int i = 0; i < batch.size(); ++i) {
            EventEnvelope out = events[i];
            InferResult r;
            r.stream_id     = out.stream_id;
            r.frame_seq     = out.frame_seq;
            r.frame_ts      = out.frame->meta.capture_ts;
            r.frame_mono_ns = out.frame->meta.capture_mono_ns;
            r.infer_ts      = infer_ts;
            r.infer_ms      = infer_ms;
            r.decode_ms     = decode_ms;
            r.model_id      = model_cfg_.id;
            if (r.frame_mono_ns != 0) {
                r.queue_latency_ms = batch_start_ns >= r.frame_mono_ns
                    ? nsToMs(batch_start_ns - r.frame_mono_ns) : 0.0;
                r.latency_ms = finish_ns >= r.frame_mono_ns
                    ? nsToMs(finish_ns - r.frame_mono_ns) : 0.0;
            } else {
                r.latency_ms = (infer_ts - r.frame_ts) * 1000.0;
            }
            if (i < static_cast<int>(decoded.size())) {
                r.detections = std::move(decoded[i]);
                const int fw = batch.is_gpu
                    ? (i < static_cast<int>(batch.gpu_frames.size()) ? batch.gpu_frames[i].width : 0)
                    : (i < static_cast<int>(batch.frames.size()) ? batch.frames[i].cols : 0);
                const int fh = batch.is_gpu
                    ? (i < static_cast<int>(batch.gpu_frames.size()) ? batch.gpu_frames[i].height : 0)
                    : (i < static_cast<int>(batch.frames.size()) ? batch.frames[i].rows : 0);
                mapDetectionsFromModelToFrame(r.detections, fw, fh, model_cfg_.input_shape);
                for (auto& d : r.detections) {
                    if (d.class_id >= 0 && d.class_id < static_cast<int>(model_cfg_.class_names.size()))
                        d.class_name = model_cfg_.class_names[d.class_id];
                }
            }
            if (r.latency_ms > 0) Metrics::get().recordE2eLatency(r.stream_id, r.latency_ms);
            out.infer_result = std::move(r);
            outputs.push_back(std::move(out));
        }
    } catch (const std::exception& e) {
        const std::string err = e.what();
        if (err.find("invalid dimensions") != std::string::npos && flush_count > 1) {
            LOG_WARN("InferEngineStage[{}]: dimension mismatch, fallback to bs=1", id_);
            fallback_to_single_infer_ = true;
            for (int i = 0; i < flush_count; ++i) {
                try {
                    Batch one;
                    std::vector<float> out_buf;
                    std::vector<std::vector<Detection>> dec;
                    const auto t0 = std::chrono::steady_clock::now();
                    const auto t1 = [&] {
                        std::lock_guard infer_lock(infer_mutex_);
                        if (!loaded_) {
                            backend_->loadModel(model_cfg_);
                            loaded_ = true;
                        }
                        one.is_gpu = events[i].frame->is_gpu;
                        one.metas.push_back(events[i].frame->meta);
                        if (one.is_gpu) one.gpu_frames.push_back(events[i].frame->gpu_buf);
                        else one.frames.push_back(events[i].frame->image);
                        backend_->infer(one, out_buf);
                        return std::chrono::steady_clock::now();
                    }();
                    const auto decode_start = std::chrono::steady_clock::now();
                    {
                        std::lock_guard infer_lock(infer_mutex_);
                        dec = decoder_->decode(out_buf.data(), 1, model_cfg_.input_shape,
                                               model_cfg_.conf_thresh, model_cfg_.nms_thresh,
                                               out_buf.size());
                    }
                    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    Metrics::get().recordInferLatency(model_cfg_.id, ms);
                    Metrics::get().recordInferBatchSize(model_cfg_.id, 1);
                    Metrics::get().incInferBatches(model_cfg_.id);
                    {
                        const uint64_t bs_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t0.time_since_epoch()).count());
                        const uint64_t cap_ns = events[i].frame->meta.capture_mono_ns;
                        if (cap_ns != 0 && bs_ns >= cap_ns)
                            Metrics::get().recordInferQueueLatency(model_cfg_.id, nsToMs(bs_ns - cap_ns));
                    }
                    const double dec_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - decode_start).count();

                    EventEnvelope ev_out = events[i];
                    InferResult r;
                    r.stream_id = ev_out.stream_id; r.frame_seq = ev_out.frame_seq;
                    r.frame_ts = ev_out.frame->meta.capture_ts;
                    r.frame_mono_ns = ev_out.frame->meta.capture_mono_ns;
                    r.infer_ts = nowEpoch(); r.infer_ms = ms; r.decode_ms = dec_ms;
                    r.model_id = model_cfg_.id;
                    {
                        const uint64_t bs_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t0.time_since_epoch()).count());
                        const uint64_t fin = nowSteadyNs();
                        if (r.frame_mono_ns != 0) {
                            r.queue_latency_ms = bs_ns >= r.frame_mono_ns
                                ? nsToMs(bs_ns - r.frame_mono_ns) : 0.0;
                            r.latency_ms = fin >= r.frame_mono_ns
                                ? nsToMs(fin - r.frame_mono_ns) : 0.0;
                        } else {
                            r.latency_ms = (r.infer_ts - r.frame_ts) * 1000.0;
                        }
                    }
                    if (!dec.empty()) {
                        r.detections = std::move(dec[0]);
                        const int fw = one.is_gpu
                            ? (!one.gpu_frames.empty() ? one.gpu_frames[0].width : 0)
                            : (!one.frames.empty() ? one.frames[0].cols : 0);
                        const int fh = one.is_gpu
                            ? (!one.gpu_frames.empty() ? one.gpu_frames[0].height : 0)
                            : (!one.frames.empty() ? one.frames[0].rows : 0);
                        mapDetectionsFromModelToFrame(r.detections, fw, fh, model_cfg_.input_shape);
                        for (auto& d : r.detections) {
                            if (d.class_id >= 0 && d.class_id < static_cast<int>(model_cfg_.class_names.size()))
                                d.class_name = model_cfg_.class_names[d.class_id];
                        }
                    }
                    if (r.latency_ms > 0) Metrics::get().recordE2eLatency(r.stream_id, r.latency_ms);
                    ev_out.infer_result = std::move(r);
                    outputs.push_back(std::move(ev_out));
                } catch (const std::exception& one_e) {
                    LOG_ERROR("InferEngineStage[{}]: fallback bs=1 failed: {}", id_, one_e.what());
                }
            }
        } else {
            LOG_ERROR("InferEngineStage[{}]: inference failed: {}", id_, err);
            outputs = std::move(events);
        }
    }
    for (auto& out : outputs) {
        emit(out);
    }
}

// Background timer: wakes every max_queue_delay_us/2, extracts frames under lock,
// then calls inferAndEmit WITHOUT the lock so downstream backpressure can't stall the queue.
void InferEngineStage::flushLoop() {
    const auto half_delay = std::chrono::microseconds(
        std::max(1000, model_cfg_.max_queue_delay_us / 2));

    while (!flush_stop_.load()) {
        std::vector<EventEnvelope> to_flush;
        EmitFn emit_copy;
        {
            std::unique_lock lock(pending_mutex_);
            flush_cv_.wait_for(lock, half_delay, [this] { return flush_stop_.load(); });
            if (flush_stop_.load()) break;

            if (!pending_events_.empty() &&
                std::chrono::steady_clock::now() >= batch_deadline_ &&
                last_emit_) {
                const int effective_bs = fallback_to_single_infer_
                    ? 1
                    : std::max(1, std::min(model_cfg_.batch_size, backend_->maxBatchSize()));
                const int fc = std::min(static_cast<int>(pending_events_.size()), effective_bs);
                to_flush = extractBatch(fc);  // pops frames, resets deadline
                emit_copy = last_emit_;
            }
        }  // pending_mutex_ released before heavy work

        if (!to_flush.empty() && emit_copy) {
            LOG_DEBUG("InferEngineStage[{}]: deadline flush bs={}", id_, to_flush.size());
            inferAndEmit(std::move(to_flush), emit_copy);
        }
    }
}

std::size_t InferEngineStage::pendingQueueSize() {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return pending_events_.size();
}

void InferEngineStage::process(const EventEnvelope& input, const EmitFn& emit) {
    std::vector<EventEnvelope> to_flush;
    {
        std::unique_lock lock(pending_mutex_);

        if (input.frame) {
            if (static_cast<int>(pending_events_.size()) >= max_pending_) {
                LOG_WARN("InferEngineStage[{}]: pending full (size={} max={}), dropping oldest",
                         id_,
                         pending_events_.size(),
                         max_pending_);
                pending_events_.pop_front();
            }
            EventEnvelope stamped = input;
            stamped.received_at_infer_ns = nowSteadyNs();
            pending_events_.push_back(std::move(stamped));
        }
        last_emit_ = emit;

        if (!pending_events_.empty()) {
            const int effective_bs = fallback_to_single_infer_
                ? 1
                : std::max(1, std::min(model_cfg_.batch_size, backend_->maxBatchSize()));
            const bool hit_batch    = static_cast<int>(pending_events_.size()) >= effective_bs;
            const bool hit_deadline = std::chrono::steady_clock::now() >= batch_deadline_;
            if (hit_batch || hit_deadline) {
                const int fc = std::min(static_cast<int>(pending_events_.size()), effective_bs);
                to_flush = extractBatch(fc);  // pops frames, resets deadline
            }
        }
    }  // pending_mutex_ released before heavy work

    if (!to_flush.empty()) {
        inferAndEmit(std::move(to_flush), emit);
    }
}

} // namespace infer
