#include "pipeline/InferWorker.h"
#include "pipeline/CascadeRouter.h"
#include "pipeline/ResultMerger.h"
#include "pipeline/stages/DetectionOverlay.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <chrono>

namespace infer {

namespace {
double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}
} // namespace

InferWorker::InferWorker(const ModelConfig&            model_cfg,
                          std::unique_ptr<IInferBackend> backend,
                          std::unique_ptr<IYOLODecoder>  decoder,
                          IPublisher&                    publisher,
                          std::shared_ptr<FrameArchiver> frame_archiver,
                          std::shared_ptr<TrackerManager> tracker_manager,
                          std::function<TrackerType(const std::string&)> tracker_type_resolver,
                          std::function<ByteTrackConfig(const std::string&)> bytetrack_config_resolver)
    : model_cfg_(model_cfg)
    , backend_(std::move(backend))
    , decoder_(std::move(decoder))
    , publisher_(publisher)
    , frame_archiver_(std::move(frame_archiver))
    , tracker_manager_(std::move(tracker_manager))
    , tracker_type_resolver_(std::move(tracker_type_resolver))
    , bytetrack_config_resolver_(std::move(bytetrack_config_resolver))
{}

InferWorker::~InferWorker() {
    stop();
}

void InferWorker::start() {
    if (running_.load()) return;
    backend_->loadModel(model_cfg_);
    stop_flag_ = false;
    thread_    = std::thread(&InferWorker::workerLoop, this);
}

void InferWorker::stop() {
    stop_flag_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    backend_->unloadModel();
    running_.store(false);
}

void InferWorker::enqueue(Batch batch) {
    std::unique_lock lock(mutex_);
    if (queue_.size() >= kMaxQueueSize) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    queue_.push(std::move(batch));
    lock.unlock();
    cv_.notify_one();
}

void InferWorker::workerLoop() {
    running_.store(true);
    LOG_INFO("InferWorker[{}]: started", model_cfg_.id);

    const InferShape shape = model_cfg_.input_shape;

    while (!stop_flag_.load()) {
        Batch batch;
        {
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(20),
                         [this] { return !queue_.empty() || stop_flag_.load(); });
            if (queue_.empty()) continue;
            batch = std::move(queue_.front());
            queue_.pop();
        }

        if (batch.empty()) continue;

        const double dequeue_ts = nowEpoch();
        const uint64_t dequeue_mono_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        try {
            auto infer_start = std::chrono::steady_clock::now();

            std::vector<float> output;
            backend_->infer(batch, output);

            auto infer_end = std::chrono::steady_clock::now();
            const uint64_t infer_end_mono_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    infer_end.time_since_epoch()).count());
            const double infer_ms = std::chrono::duration<double, std::milli>(
                infer_end - infer_start).count();
            Metrics::get().recordInferLatency(model_cfg_.id, infer_ms);
            Metrics::get().recordInferBatchSize(model_cfg_.id, static_cast<int>(batch.size()));
            Metrics::get().incInferBatches(model_cfg_.id);

            auto decode_start = std::chrono::steady_clock::now();
            auto per_image = decoder_->decode(
                output.data(), batch.size(), shape,
                model_cfg_.conf_thresh, model_cfg_.nms_thresh);
            const double decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();

            double infer_ts = nowEpoch();
            for (int i = 0; i < batch.size(); ++i) {
                InferResult r;
                r.stream_id       = batch.metas[i].stream_id;
                r.frame_ts        = batch.metas[i].capture_ts;
                r.frame_mono_ns   = batch.metas[i].capture_mono_ns;
                r.frame_seq       = batch.metas[i].frame_seq;
                r.infer_ts        = infer_ts;
                if (r.frame_mono_ns != 0) {
                    r.queue_latency_ms = (static_cast<double>(dequeue_mono_ns - r.frame_mono_ns)) / 1e6;
                    r.latency_ms       = (static_cast<double>(infer_end_mono_ns - r.frame_mono_ns)) / 1e6;
                } else {
                    // Fallback: compute using epoch seconds (can be affected by system clock jumps).
                    r.queue_latency_ms = (dequeue_ts - r.frame_ts) * 1000.0;
                    r.latency_ms       = (infer_ts - r.frame_ts) * 1000.0;
                }
                r.infer_ms        = infer_ms;
                r.decode_ms       = decode_ms;
                r.model_id        = model_cfg_.id;
                r.detections = std::move(per_image[i]);
                const int fw = batch.is_gpu
                    ? (i < static_cast<int>(batch.gpu_frames.size()) ? batch.gpu_frames[i].width : 0)
                    : (i < static_cast<int>(batch.frames.size()) ? batch.frames[i].cols : 0);
                const int fh = batch.is_gpu
                    ? (i < static_cast<int>(batch.gpu_frames.size()) ? batch.gpu_frames[i].height : 0)
                    : (i < static_cast<int>(batch.frames.size()) ? batch.frames[i].rows : 0);
                mapDetectionsFromModelToFrame(r.detections, fw, fh, shape);
                if (frame_archiver_ && !batch.is_gpu && i < static_cast<int>(batch.frames.size())) {
                    const auto ar = frame_archiver_->submit(batch.metas[i], &batch.frames[i]);
                    r.frame_local_path  = ar.local_path;
                    r.frame_url         = ar.frame_url;
                    r.frame_upload_state = ar.upload_state;
                } else {
                    r.frame_upload_state = "disabled";
                }

                Metrics::get().recordE2eLatency(r.stream_id, r.latency_ms);

                // Resolve class names if configured
                for (auto& d : r.detections) {
                    if (d.class_id < static_cast<int>(model_cfg_.class_names.size())) {
                        d.class_name = model_cfg_.class_names[d.class_id];
                    }
                }

                if (tracker_manager_ && tracker_type_resolver_) {
                    const TrackerType tracker_type = tracker_type_resolver_(r.stream_id);
                    const ByteTrackConfig bt_cfg = bytetrack_config_resolver_
                        ? bytetrack_config_resolver_(r.stream_id)
                        : ByteTrackConfig{};
                    tracker_manager_->apply(r.stream_id, tracker_type, bt_cfg, r.frame_seq, r.detections);
                }

                if (cascade_router_) {
                    // Primary model with cascade: route to secondary and let
                    // ResultMerger publish after secondary attributes arrive.
                    const GpuBuffer*  gpu_frame = batch.is_gpu && i < static_cast<int>(batch.gpu_frames.size())
                                                  ? &batch.gpu_frames[i] : nullptr;
                    const cv::Mat*    cpu_frame = !batch.is_gpu && i < static_cast<int>(batch.frames.size())
                                                  ? &batch.frames[i] : nullptr;
                    cascade_router_->route(r, i, gpu_frame, cpu_frame);
                    // Flush expired entries after each frame
                    if (result_merger_) result_merger_->flushExpired();
                } else {
                    publisher_.publish(std::move(r));
                }
            }
            processed_batches_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            LOG_ERROR("InferWorker[{}]: infer failed: {}", model_cfg_.id, e.what());
        }
    }

    LOG_INFO("InferWorker[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

} // namespace infer
