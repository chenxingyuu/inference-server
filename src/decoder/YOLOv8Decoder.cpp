#include "decoder/YOLOv8Decoder.h"
#include <algorithm>

namespace infer {

YOLOv8Decoder::YOLOv8Decoder(int num_classes)
    : num_classes_(num_classes) {}

// output layout: [4 + num_classes, num_anchors]  (transposed from TRT export)
std::vector<Detection> YOLOv8Decoder::decodeSingle(
    const float* data, int num_anchors,
    float conf_thresh, float nms_thresh,
    int img_w, int img_h)
{
    const int rows = 4 + num_classes_;
    std::vector<Detection> dets;

    for (int a = 0; a < num_anchors; ++a) {
        // Find best class
        int   best_cls  = 0;
        float best_prob = 0.f;
        for (int c = 0; c < num_classes_; ++c) {
            float prob = data[(4 + c) * num_anchors + a];
            if (prob > best_prob) { best_prob = prob; best_cls = c; }
        }
        if (best_prob < conf_thresh) continue;

        float x1 = data[0 * num_anchors + a];
        float y1 = data[1 * num_anchors + a];
        float x2 = data[2 * num_anchors + a];
        float y2 = data[3 * num_anchors + a];

        Detection d;
        d.class_id   = best_cls;
        d.confidence = best_prob;
        d.bbox       = {std::max(0.f, x1),
                        std::max(0.f, y1),
                        std::min(static_cast<float>(img_w), x2),
                        std::min(static_cast<float>(img_h), y2)};
        dets.push_back(d);
    }

    auto keep = nonMaxSuppression(dets, nms_thresh);
    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) result.push_back(dets[idx]);
    return result;
}

std::vector<std::vector<Detection>> YOLOv8Decoder::decode(
    const float*      output,
    int               batch_size,
    const InferShape& shape,
    float             conf_thresh,
    float             nms_thresh)
{
    const int num_anchors = 8400;
    const int rows        = 4 + num_classes_;
    const int img_w       = shape.width;
    const int img_h       = shape.height;

    std::vector<std::vector<Detection>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        const float* data = output + b * rows * num_anchors;
        results[b] = decodeSingle(data, num_anchors,
                                  conf_thresh, nms_thresh, img_w, img_h);
    }
    return results;
}

} // namespace infer
