#pragma once

#include "decoder/YOLOv8Decoder.h"

namespace infer {

// YOLO11 uses the same output format as YOLOv8 (direct box regression).
// Inherits YOLOv8Decoder and only overrides version().
class YOLO11Decoder : public YOLOv8Decoder {
public:
    explicit YOLO11Decoder(int num_classes = 80)
        : YOLOv8Decoder(num_classes) {}

    YOLOVersion version() const override { return YOLOVersion::v11; }
};

} // namespace infer
