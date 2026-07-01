#include "pipeline/InferWorker.h"
#include "pipeline/CascadeRouter.h"
#include "pipeline/ResultMerger.h"
#include "pipeline/stages/DetectionOverlay.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <chrono>
#include <thread>

#ifdef BUILD_TRT_BACKEND
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#endif

namespace infer {

namespace {
double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

#ifdef BUILD_TRT_BACKEND
// Download NV12 GPU buffer to host and convert to BGR cv::Mat for archiving.
cv::Mat nv12GpuToBgr(const GpuBuffer& gb) {
    if (!gb.y_data || gb.width <= 0 || gb.height <= 0) return {};
    const int w = gb.width;
    const int h = gb.height;
    const size_t y_bytes  = static_cast<size_t>(w) * h;
    const size_t uv_bytes = static_cast<size_t>(w) * (h / 2);
    cv::Mat nv12(h + h / 2, w, CV_8UC1);
    cudaMemcpy(nv12.data,             gb.y_data,  y_bytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(nv12.data + y_bytes,   gb.uv_data, uv_bytes, cudaMemcpyDeviceToHost);
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}
#endif
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
    if (state_.load() == WorkerState::STOPPED) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("InferWorker[{}]: dropped batch (worker STOPPED)", model_cfg_.id);
        return;
    }
    std::unique_lock lock(mutex_);
    if (queue_.size() >= kMaxQueueSize) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("InferWorker[{}]: dropped batch (queue full, depth={})", model_cfg_.id, queue_.size());
        return;
    }
    queue_.push_back(std::move(batch));
    lock.unlock();
    cv_.notify_one();
}

void InferWorker::enqueueHead(Batch batch) {
    std::unique_lock lock(mutex_);
    if (queue_.size() >= kMaxQueueSize) {
        // Even re-queue dropped under extreme pressure
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    queue_.push_front(std::move(batch));
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
            queue_.pop_front();
        }

        if (batch.empty()) continue;

        LOG_DEBUG("InferWorker[{}]: dequeued batch_size={} queue_remaining={}",
                  model_cfg_.id, batch.size(), queue_.size());

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
            Metrics::get().recordInferBatchSize(model_cfg_.id, batch.size());
            Metrics::get().incInferBatches(model_cfg_.id);
            LOG_DEBUG("InferWorker[{}]: infer done bs={} infer_ms={:.1f}",
                      model_cfg_.id, batch.size(), infer_ms);

            auto decode_start = std::chrono::steady_clock::now();
            auto per_image = decoder_->decode(
                output.data(), batch.size(), shape,
                model_cfg_.conf_thresh, model_cfg_.nms_thresh,
                output.size());
            const double decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            LOG_DEBUG("InferWorker[{}]: decode done bs={} decode_ms={:.1f}",
                      model_cfg_.id, batch.size(), decode_ms);
            if (per_image.size() < static_cast<size_t>(batch.size())) {
                LOG_ERROR("InferWorker[{}]: decode result size mismatch, batch_size={}, decoded={}",
                          model_cfg_.id, batch.size(), per_image.size());
                continue;
            }
            if (batch.metas.size() < static_cast<size_t>(batch.size())) {
                LOG_ERROR("InferWorker[{}]: batch metadata size mismatch, batch_size={}, metas={}",
                          model_cfg_.id, batch.size(), batch.metas.size());
                continue;
            }

            double infer_ts = nowEpoch();
            for (int i = 0; i < batch.size(); ++i) {
                InferResult r;
                r.stream_id       = batch.metas[i].stream_id;
                r.frame_ts        = batch.metas[i].capture_ts;
                r.frame_mono_ns   = batch.metas[i].capture_mono_ns;
                r.frame_seq       = batch.metas[i].frame_seq;
                r.infer_ts        = infer_ts;
                if (r.frame_mono_ns != 0) {
                    r.queue_latency_ms = dequeue_mono_ns >= r.frame_mono_ns
                        ? (static_cast<double>(dequeue_mono_ns - r.frame_mono_ns)) / 1e6 : 0.0;
                    r.latency_ms = infer_end_mono_ns >= r.frame_mono_ns
                        ? (static_cast<double>(infer_end_mono_ns - r.frame_mono_ns)) / 1e6 : 0.0;
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
                if (frame_archiver_) {
                    FrameArchiveResult ar;
                    if (!batch.is_gpu && i < static_cast<int>(batch.frames.size())) {
                        ar = frame_archiver_->submit(batch.metas[i], &batch.frames[i]);
                    }
#ifdef BUILD_TRT_BACKEND
                    else if (batch.is_gpu && i < static_cast<int>(batch.gpu_frames.size())) {
                        cv::Mat bgr = nv12GpuToBgr(batch.gpu_frames[i]);
                        ar = frame_archiver_->submit(batch.metas[i], bgr.empty() ? nullptr : &bgr);
                    }
#endif
                    r.frame_local_path   = ar.local_path;
                    r.frame_url.clear();
                    r.frame_upload_state = ar.upload_state;
                    LOG_DEBUG("InferWorker[{}]: archive stream={} state={} path={}",
                              model_cfg_.id, r.stream_id, ar.upload_state,
                              ar.local_path.empty() ? "(skipped)" : ar.local_path);
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
                    LOG_DEBUG("InferWorker[{}]: tracker applied stream={} dets={}",
                              model_cfg_.id, r.stream_id, r.detections.size());
                }

                if (cascade_router_) {
                    // Primary model with cascade: route to secondary and let
                    // ResultMerger publish after secondary attributes arrive.
                    const GpuBuffer*  gpu_frame = batch.is_gpu && i < static_cast<int>(batch.gpu_frames.size())
                                                  ? &batch.gpu_frames[i] : nullptr;
                    const cv::Mat*    cpu_frame = !batch.is_gpu && i < static_cast<int>(batch.frames.size())
                                                  ? &batch.frames[i] : nullptr;
                    LOG_DEBUG("InferWorker[{}]: cascade route stream={} dets={}",
                              model_cfg_.id, r.stream_id, r.detections.size());
                    cascade_router_->route(r, i, gpu_frame, cpu_frame);
                    // Flush expired entries after each frame
                    if (result_merger_) result_merger_->flushExpired();
                } else {
                    LOG_DEBUG("InferWorker[{}]: publish stream={} dets={} latency_ms={:.1f}",
                              model_cfg_.id, r.stream_id, r.detections.size(), r.latency_ms);
                    publisher_.publish(std::move(r));
                }
            }
            const uint64_t total = processed_batches_.fetch_add(1, std::memory_order_relaxed) + 1;
            batch_policy_.onSuccess();
            Metrics::get().setInferBatchSizeCurrent(model_cfg_.id, batch_policy_.current_max);
            if (total % 200 == 0) {
                LOG_INFO("InferWorker[{}]: stats processed={} dropped={} policy_max={}",
                         model_cfg_.id, total,
                         dropped_batches_.load(std::memory_order_relaxed),
                         batch_policy_.current_max);
            }
        } catch (const GpuMemoryPressureException& e) {
            LOG_WARN("InferWorker[{}]: GPU memory pressure: {}", model_cfg_.id, e.what());
            batch_policy_.onOOM();
            Metrics::get().incGpuOom(model_cfg_.id);
            Metrics::get().setInferBatchSizeCurrent(model_cfg_.id, batch_policy_.current_max);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            enqueueHead(std::move(batch));
        } catch (const GpuFaultException& e) {
            LOG_ERROR("InferWorker[{}]: GPU fault ({}): {}",
                      model_cfg_.id, static_cast<int>(e.fault_type), e.what());
            Metrics::get().incGpuEngineFault(model_cfg_.id);
            recoverFromFault(e.fault_type);
        } catch (const std::exception& e) {
            LOG_ERROR("InferWorker[{}]: infer failed: {}", model_cfg_.id, e.what());
        }
    }

    LOG_INFO("InferWorker[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

void InferWorker::recoverFromFault(GpuFaultType fault) {
    state_.store(WorkerState::RECOVERING);
    Metrics::get().setInferWorkerState(model_cfg_.id, 1);
    LOG_WARN("InferWorker[{}]: starting recovery (fault={})",
             model_cfg_.id, static_cast<int>(fault));

    // CONTEXT_LOST requires device reset; other faults only need engine reload.
    if (fault == GpuFaultType::CONTEXT_LOST) {
#ifdef BUILD_TRT_BACKEND
        cudaDeviceReset();
#endif
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
        try {
            backend_->unloadModel();
            backend_->loadModel(model_cfg_);
            state_.store(WorkerState::RUNNING);
            Metrics::get().setInferWorkerState(model_cfg_.id, 0);
            LOG_INFO("InferWorker[{}]: recovered (attempt {})", model_cfg_.id, attempt + 1);
            return;
        } catch (const std::exception& ex) {
            LOG_ERROR("InferWorker[{}]: recovery attempt {} failed: {}",
                      model_cfg_.id, attempt + 1, ex.what());
            std::this_thread::sleep_for(std::chrono::seconds(1 << attempt)); // 1s, 2s, 4s
        }
    }

    state_.store(WorkerState::STOPPED);
    Metrics::get().setInferWorkerState(model_cfg_.id, 2);
    LOG_CRITICAL("InferWorker[{}]: unrecoverable after 3 attempts — manual intervention required",
                 model_cfg_.id);
}

} // namespace infer
