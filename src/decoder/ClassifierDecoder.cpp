#include "decoder/ClassifierDecoder.h"
#include <stdexcept>

namespace infer {

ClassifierDecoder::ClassifierDecoder(const ModelConfig& cfg)
    : num_classes_(cfg.num_classes)
    , class_names_(cfg.class_names)
{
    if (num_classes_ <= 0)
        throw std::runtime_error("ClassifierDecoder: num_classes must be > 0");
}

std::vector<std::vector<Detection>> ClassifierDecoder::decode(
        const float*      output,
        int               batch_size,
        const InferShape& /*shape*/,
        float             conf_thresh,
        float             /*nms_thresh*/,
        size_t            /*output_count*/) {
    std::vector<std::vector<Detection>> results(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        const float* row = output + b * num_classes_;

        int   best_class = 0;
        float best_score = row[0];
        for (int c = 1; c < num_classes_; ++c) {
            if (row[c] > best_score) {
                best_score = row[c];
                best_class = c;
            }
        }

        if (best_score < conf_thresh) continue;

        Detection d;
        d.class_id   = best_class;
        d.confidence = best_score;
        if (best_class < static_cast<int>(class_names_.size())) {
            d.class_name = class_names_[best_class];
        } else {
            d.class_name = std::to_string(best_class);
        }
        results[b].push_back(std::move(d));
    }

    return results;
}

} // namespace infer
