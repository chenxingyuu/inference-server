#pragma once

#include "decoder/IYOLODecoder.h"
#include <vector>

namespace infer {

// YOLOv5 output layout: [batch, num_anchors_total, 5 + num_classes]
//   anchor_total = (80x80 + 40x40 + 20x20) × 3 = 25200 for 640-input
//   each prediction: [cx, cy, w, h, obj_conf, cls0..clsN]  (normalised or pixel)
class YOLOv5Decoder : public IYOLODecoder {
public:
    explicit YOLOv5Decoder(int num_classes = 80);

    std::vector<std::vector<Detection>> decode(
        const float*      output,
        int               batch_size,
        const InferShape& shape,
        float             conf_thresh,
        float             nms_thresh,
        size_t            output_count = 0
    ) override;

    YOLOVersion version() const override { return YOLOVersion::v5; }

private:
    std::vector<Detection> decodeSingle(
        const float* data, int num_preds, int stride,
        float conf_thresh, float nms_thresh,
        int img_w, int img_h);

    int num_classes_;
};

} // namespace infer
