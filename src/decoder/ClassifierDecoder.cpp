#include "decoder/ClassifierDecoder.h"
#include <algorithm>
#include <cmath>
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

        results[b].push_back(prediction(best_class, best_score));
    }

    return results;
}

Detection ClassifierDecoder::prediction(int class_id, float score) const {
    Detection result;
    result.class_id = class_id;
    result.confidence = score;
    result.class_name = class_id < static_cast<int>(class_names_.size())
        ? class_names_[class_id] : std::to_string(class_id);
    return result;
}

std::vector<Detection> ClassifierDecoder::rank(
        const float* scores, size_t count, float threshold, size_t limit) const {
    if (count < static_cast<size_t>(num_classes_))
        throw std::runtime_error("classifier output contains fewer scores than configured classes");

    std::vector<int> class_ids;
    for (int id = 0; id < num_classes_; ++id) {
        if (!std::isfinite(scores[id])) throw std::runtime_error("classifier output contains an invalid score");
        if (scores[id] >= threshold) class_ids.push_back(id);
    }
    const auto size = std::min(limit, class_ids.size());
    std::partial_sort(class_ids.begin(), class_ids.begin() + size, class_ids.end(),
                      [&](int a, int b) { return scores[a] == scores[b] ? a < b : scores[a] > scores[b]; });
    std::vector<Detection> results;
    results.reserve(size);
    for (size_t i = 0; i < size; ++i) results.push_back(prediction(class_ids[i], scores[class_ids[i]]));
    return results;
}

} // namespace infer
