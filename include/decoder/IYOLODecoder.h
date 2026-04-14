#pragma once

#include "common/Types.h"
#include <vector>

namespace infer {

class IYOLODecoder {
public:
    virtual ~IYOLODecoder() = default;

    // Decode raw model output into per-image detection lists.
    // output    : flat float buffer from IInferBackend::infer()
    // batch_size: number of images in the batch
    // shape     : model input shape (for coordinate scaling)
    // conf_thresh: minimum objectness × class score
    // nms_thresh : IoU threshold for NMS
    // Returns vector of length batch_size; each element is detections for one image.
    virtual std::vector<std::vector<Detection>> decode(
        const float*   output,
        int            batch_size,
        const InferShape& shape,
        float          conf_thresh,
        float          nms_thresh
    ) = 0;

    virtual YOLOVersion version() const = 0;
};

// NMS utility shared by all decoders
std::vector<int> nonMaxSuppression(
    const std::vector<Detection>& dets,
    float iou_thresh);

float iou(const BBox& a, const BBox& b);

} // namespace infer
