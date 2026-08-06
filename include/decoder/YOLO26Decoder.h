#pragma once

#include "decoder/IYOLODecoder.h"

namespace infer {

// YOLO26 is natively end-to-end (NMS-free). The exported model emits a fixed
// output layout: [batch, num_dets, 6]  (row-major, num_dets = 300 by default).
//   Each row: [x1, y1, x2, y2, confidence, class_id]
//     - x1,y1,x2,y2 : absolute pixel corners (xyxy) in model-input resolution
//     - confidence  : detection score; padding rows are zero-filled
//     - class_id    : emitted as float, rounded to int
// The head already de-duplicates predictions, so decode() runs no NMS; the
// nms_thresh argument is accepted for interface parity but ignored.
class YOLO26Decoder : public IYOLODecoder {
public:
    explicit YOLO26Decoder(int num_classes = 80)
        : num_classes_(num_classes) {}

    std::vector<std::vector<Detection>> decode(
        const float*      output,
        int               batch_size,
        const InferShape& shape,
        float             conf_thresh,
        float             nms_thresh,
        size_t            output_count = 0
    ) override;

    YOLOVersion version() const override { return YOLOVersion::v26; }

private:
    [[maybe_unused]] int num_classes_;
};

} // namespace infer
