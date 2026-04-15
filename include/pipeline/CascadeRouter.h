#pragma once

#include "common/Types.h"
#include "common/Config.h"
#include "pipeline/ResultMerger.h"
#include <memory>
#include <functional>
#include <unordered_map>

namespace infer {

class InferWorkerGroup;

// Routes primary InferResults to secondary (classifier) models.
// Inspired by DeepStream's primary GIE → secondary GIE cascade.
//
// For each primary InferResult:
//   1. Find detections whose class_id matches trigger_classes
//   2. Crop the ROI from the original frame (GPU or CPU)
//   3. Assemble a mini-batch and enqueue to the secondary InferWorkerGroup
//   4. Register the primary result in ResultMerger to await secondary results
//
// When no secondary results are pending (or on timeout), ResultMerger publishes.
class CascadeRouter {
public:
    // secondary_groups: model_id → InferWorkerGroup for that secondary model
    CascadeRouter(const ModelConfig&                                     primary_cfg,
                  std::unordered_map<std::string, InferWorkerGroup*>     secondary_groups,
                  ResultMerger&                                          merger);

    // Route one primary frame result.
    // `frame_idx` indexes into the original batch (needed to access the frame data).
    // `original_frame_gpu`: the GPU NV12 buffer for this frame (may be null for CPU path)
    // `original_frame_cpu`: the CPU cv::Mat for this frame (may be empty for GPU path)
    void route(const InferResult& result,
               int                frame_idx,
               const GpuBuffer*   original_frame_gpu,
               const cv::Mat*     original_frame_cpu);

private:
    // Clip and expand a bounding box, returning pixel-space coordinates.
    struct CropRect { int x0, y0, x1, y1; };
    CropRect computeCrop(const BBox& bbox, float expand,
                         int img_w, int img_h) const;

    ModelConfig                                          primary_cfg_;
    std::unordered_map<std::string, InferWorkerGroup*>   secondary_groups_;
    ResultMerger&                                        merger_;
};

} // namespace infer
