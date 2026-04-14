#pragma once

#include "decoder/IYOLODecoder.h"

namespace infer {

// YOLOv8 output layout: [batch, 4 + num_classes, num_anchors]
//   num_anchors = 8400 for 640-input (80x80 + 40x40 + 20x20)
//   first 4 rows: [x1, y1, x2, y2]  (absolute coords, no sigmoid)
//   remaining rows: class probabilities (softmax already applied by model)
class YOLOv8Decoder : public IYOLODecoder {
public:
    explicit YOLOv8Decoder(int num_classes = 80);

    std::vector<std::vector<Detection>> decode(
        const float*      output,
        int               batch_size,
        const InferShape& shape,
        float             conf_thresh,
        float             nms_thresh
    ) override;

    YOLOVersion version() const override { return YOLOVersion::v8; }

protected:
    std::vector<Detection> decodeSingle(
        const float* data, int num_anchors,
        float conf_thresh, float nms_thresh,
        int img_w, int img_h);

    int num_classes_;
};

} // namespace infer
