#include "decoder/YOLOv8Decoder.h"
#include "common/Logger.h"
#include <algorithm>

namespace infer {

YOLOv8Decoder::YOLOv8Decoder(int num_classes)
    : num_classes_(num_classes) {}

// output layout: [4 + num_classes, num_anchors]
// box coords are cx/cy/w/h (ultralytics raw ONNX export, no built-in NMS)
std::vector<Detection> YOLOv8Decoder::decodeSingle(
    const float* data, int num_anchors,
    float conf_thresh, float nms_thresh,
    int img_w, int img_h)
{
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

        float cx = data[0 * num_anchors + a];
        float cy = data[1 * num_anchors + a];
        float w  = data[2 * num_anchors + a];
        float h  = data[3 * num_anchors + a];
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

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
    float             nms_thresh,
    size_t            output_count)
{
    const int rows = 4 + num_classes_;
    int num_anchors = 8400;
    if (output_count > 0 && batch_size > 0) {
        num_anchors = static_cast<int>(output_count / static_cast<size_t>(batch_size) / rows);
    }
    const int img_w       = shape.width;
    const int img_h       = shape.height;

    std::vector<std::vector<Detection>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        const float* data = output + b * rows * num_anchors;
        results[b] = decodeSingle(data, num_anchors,
                                  conf_thresh, nms_thresh, img_w, img_h);
    }
    int total_dets = 0;
    for (const auto& r : results) total_dets += static_cast<int>(r.size());
    LOG_DEBUG("YOLOv8: batch_size={} dets={}", batch_size, total_dets);
    return results;
}

} // namespace infer
