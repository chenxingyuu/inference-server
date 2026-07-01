#include "decoder/YOLOv5Decoder.h"
#include <cmath>
#include <algorithm>

namespace infer {

YOLOv5Decoder::YOLOv5Decoder(int num_classes)
    : num_classes_(num_classes) {}

std::vector<Detection> YOLOv5Decoder::decodeSingle(
    const float* data, int num_preds, int /*stride*/,
    float conf_thresh, float nms_thresh,
    int img_w, int img_h)
{
    // Each prediction: [cx, cy, w, h, obj_conf, cls0..clsN-1]  (all post-sigmoid)
    const int step = 5 + num_classes_;
    std::vector<Detection> dets;

    for (int i = 0; i < num_preds; ++i) {
        const float* p      = data + i * step;
        float        obj    = p[4];
        if (obj < conf_thresh) continue;

        int   best_cls  = 0;
        float best_prob = 0.f;
        for (int c = 0; c < num_classes_; ++c) {
            float prob = obj * p[5 + c];
            if (prob > best_prob) { best_prob = prob; best_cls = c; }
        }
        if (best_prob < conf_thresh) continue;

        float cx = p[0] * img_w;
        float cy = p[1] * img_h;
        float bw = p[2] * img_w;
        float bh = p[3] * img_h;

        Detection d;
        d.class_id   = best_cls;
        d.confidence = best_prob;
        d.bbox       = {cx - bw / 2, cy - bh / 2,
                        cx + bw / 2, cy + bh / 2};
        d.bbox.x1 = std::max(0.f, d.bbox.x1);
        d.bbox.y1 = std::max(0.f, d.bbox.y1);
        d.bbox.x2 = std::min(static_cast<float>(img_w), d.bbox.x2);
        d.bbox.y2 = std::min(static_cast<float>(img_h), d.bbox.y2);
        dets.push_back(d);
    }

    auto keep = nonMaxSuppression(dets, nms_thresh);
    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) result.push_back(dets[idx]);
    return result;
}

std::vector<std::vector<Detection>> YOLOv5Decoder::decode(
    const float*      output,
    int               batch_size,
    const InferShape& shape,
    float             conf_thresh,
    float             nms_thresh,
    size_t            output_count)
{
    // YOLOv5 exports: output[batch, num_preds, 5+cls]
    const int step = 5 + num_classes_;
    int num_preds  = 25200;
    if (output_count > 0 && batch_size > 0) {
        num_preds = static_cast<int>(output_count / static_cast<size_t>(batch_size) / step);
    }
    const int img_w     = shape.width;
    const int img_h     = shape.height;

    std::vector<std::vector<Detection>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        const float* data = output + b * num_preds * step;
        results[b] = decodeSingle(data, num_preds, step,
                                  conf_thresh, nms_thresh, img_w, img_h);
    }
    return results;
}

} // namespace infer
