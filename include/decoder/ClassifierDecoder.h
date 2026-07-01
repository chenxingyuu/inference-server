#pragma once

#include "decoder/IYOLODecoder.h"
#include "common/Config.h"
#include <vector>
#include <string>

namespace infer {

// Decodes a classifier output: [batch, num_classes] float tensor → top-1 Detection.
// Implements IYOLODecoder so it can be used inside the standard InferWorker pipeline.
//
// Each sample in the batch produces at most one Detection:
//   - class_id / class_name: argmax of scores
//   - confidence: raw score value
//   - bbox: zeroed (not applicable for classifiers)
//
// Used by the cascade secondary model path; the result is routed via
// AttributePublisher into ResultMerger instead of being published to Kafka.
class ClassifierDecoder : public IYOLODecoder {
public:
    explicit ClassifierDecoder(const ModelConfig& cfg);

    // IYOLODecoder: returns ≤1 detection per sample (top-1 class, if score ≥ conf_thresh).
    // conf_thresh is used to filter low-confidence predictions.
    // nms_thresh is ignored (not applicable).
    std::vector<std::vector<Detection>> decode(
        const float*      output,
        int               batch_size,
        const InferShape& shape,
        float             conf_thresh,
        float             nms_thresh,
        size_t            output_count = 0) override;

    YOLOVersion version() const override { return YOLOVersion::Unknown; }

private:
    int                      num_classes_{1};
    std::vector<std::string> class_names_;
};

} // namespace infer
