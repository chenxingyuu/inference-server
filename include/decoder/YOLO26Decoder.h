#pragma once

#include "decoder/IYOLODecoder.h"

namespace infer {

// YOLO26 decoder — reserved interface.
// Output format is not yet finalized; this stub ensures the factory compiles.
// Fill in decode() once the YOLO26 architecture is published.
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
