#include "pipeline/CascadeRouter.h"
#include "pipeline/InferWorkerGroup.h"
#include "common/Logger.h"

#include <algorithm>
#include <cmath>

#ifdef BUILD_TRT_BACKEND
#include "cuda/CudaPreprocess.h"
#include <cuda_runtime_api.h>
#endif

#include <opencv2/imgproc.hpp>

namespace infer {

CascadeRouter::CascadeRouter(
        const ModelConfig&                                  primary_cfg,
        std::unordered_map<std::string, InferWorkerGroup*>  secondary_groups,
        ResultMerger&                                       merger)
    : primary_cfg_(primary_cfg)
    , secondary_groups_(std::move(secondary_groups))
    , merger_(merger)
{}

CascadeRouter::CropRect CascadeRouter::computeCrop(
        const BBox& bbox, float expand, int img_w, int img_h) const {
    const float dx = (bbox.x2 - bbox.x1) * expand;
    const float dy = (bbox.y2 - bbox.y1) * expand;

    CropRect r;
    r.x0 = static_cast<int>(std::max(0.f, bbox.x1 - dx));
    r.y0 = static_cast<int>(std::max(0.f, bbox.y1 - dy));
    r.x1 = static_cast<int>(std::min(static_cast<float>(img_w), bbox.x2 + dx));
    r.y1 = static_cast<int>(std::min(static_cast<float>(img_h), bbox.y2 + dy));

    // Ensure minimum 1×1
    if (r.x1 <= r.x0) r.x1 = r.x0 + 1;
    if (r.y1 <= r.y0) r.y1 = r.y0 + 1;
    return r;
}

void CascadeRouter::route(const InferResult& result,
                           int                /*frame_idx*/,
                           const GpuBuffer*   original_frame_gpu,
                           const cv::Mat*     original_frame_cpu) {
    const int img_w = original_frame_gpu ? original_frame_gpu->width
                    : (original_frame_cpu ? original_frame_cpu->cols : 0);
    const int img_h = original_frame_gpu ? original_frame_gpu->height
                    : (original_frame_cpu ? original_frame_cpu->rows : 0);

    // Phase 1: Count the total number of secondary inference jobs that will be
    // dispatched across ALL cascade configs for this frame. We must call
    // registerPrimary() with this total BEFORE any enqueue() to guarantee the
    // pending entry exists when the first secondary result arrives (TOCTOU fix).
    // A single registerPrimary() call also prevents duplicate-key collisions
    // when the primary model has more than one cascade config.
    int total_expected = 0;
    if (img_w > 0 && img_h > 0) {
        for (const auto& cas : primary_cfg_.cascade) {
            if (!secondary_groups_.count(cas.model_id)) continue;
            for (const auto& det : result.detections) {
                const bool triggered = cas.trigger_classes.empty() ||
                    std::find(cas.trigger_classes.begin(), cas.trigger_classes.end(),
                              det.class_id) != cas.trigger_classes.end();
                if (triggered) ++total_expected;
            }
        }
    }

    // Register primary result BEFORE dispatching crops.
    // If total_expected == 0, registerPrimary publishes immediately.
    merger_.registerPrimary(InferResult{result}, total_expected);

    if (total_expected == 0) return;

    // Phase 2: Dispatch crop mini-batches to each secondary worker group.
    for (const auto& cas : primary_cfg_.cascade) {
        auto group_it = secondary_groups_.find(cas.model_id);
        if (group_it == secondary_groups_.end()) {
            LOG_WARN("CascadeRouter: secondary model {} not found", cas.model_id);
            continue;
        }
        InferWorkerGroup* secondary = group_it->second;

        for (int i = 0; i < static_cast<int>(result.detections.size()); ++i) {
            const auto& det = result.detections[i];
            const bool triggered = cas.trigger_classes.empty() ||
                std::find(cas.trigger_classes.begin(), cas.trigger_classes.end(),
                          det.class_id) != cas.trigger_classes.end();
            if (!triggered) continue;

            auto crop = computeCrop(det.bbox, cas.crop_expand, img_w, img_h);

            if (original_frame_gpu) {
#ifdef BUILD_TRT_BACKEND
                // GPU path: crop + resize to secondary model input size.
                // TODO: query secondary model config for actual input_size.
                const int sec_h = 112, sec_w = 112;
                const size_t crop_bytes = 3 * sec_h * sec_w * sizeof(float);

                void* crop_device = nullptr;
                cudaMalloc(&crop_device, crop_bytes);

                // Use the frame's CUDA stream so the crop kernel sees the
                // fully-decoded NV12 data (avoids default-stream aliasing).
                cudaStream_t frame_stream = static_cast<cudaStream_t>(
                    original_frame_gpu->cuda_stream);
                cuda::cropAndResizeNV12ToCHW(
                    static_cast<const uint8_t*>(original_frame_gpu->y_data),
                    static_cast<const uint8_t*>(original_frame_gpu->uv_data),
                    original_frame_gpu->height, original_frame_gpu->width,
                    crop.x0, crop.y0, crop.x1, crop.y1,
                    static_cast<float*>(crop_device), sec_h, sec_w,
                    0.0f, 1.0f / 255.0f, frame_stream);
                cudaStreamSynchronize(frame_stream);

                // GpuBuffer carries the float CHW device memory. frame_ref
                // owns the allocation and frees it when the batch is consumed.
                GpuBuffer crop_gb;
                crop_gb.y_data    = crop_device;
                crop_gb.width     = sec_w;
                crop_gb.height    = sec_h;
                crop_gb.frame_ref = std::shared_ptr<void>(crop_device,
                    [](void* p) { cudaFree(p); });

                Batch mini_batch;
                mini_batch.gpu_frames.push_back(std::move(crop_gb));
                mini_batch.metas.push_back(StreamMeta{
                    result.stream_id, cas.model_id,
                    result.frame_ts, static_cast<uint64_t>(i), img_w, img_h});
                mini_batch.is_gpu = true;
                secondary->enqueue(std::move(mini_batch));
#endif
            } else if (original_frame_cpu && !original_frame_cpu->empty()) {
                // CPU path: crop with OpenCV, resize to secondary model input size.
                // TODO: query secondary model config for actual input_size.
                cv::Rect roi_rect(crop.x0, crop.y0,
                                  crop.x1 - crop.x0, crop.y1 - crop.y0);
                cv::Mat roi_crop = (*original_frame_cpu)(roi_rect).clone();
                cv::Mat roi_resized;
                cv::resize(roi_crop, roi_resized, {112, 112});

                StreamMeta sm;
                sm.stream_id   = result.stream_id;
                sm.model_id    = cas.model_id;
                sm.capture_ts  = result.frame_ts;
                sm.frame_seq   = static_cast<uint64_t>(i);  // det_idx
                sm.orig_width  = img_w;
                sm.orig_height = img_h;

                Batch mini_batch;
                mini_batch.frames.push_back(std::move(roi_resized));
                mini_batch.metas.push_back(std::move(sm));
                mini_batch.is_gpu = false;
                secondary->enqueue(std::move(mini_batch));
            }
        }
    }
}

} // namespace infer
