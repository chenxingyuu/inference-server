#include "decoder/YOLO26Decoder.h"
#include "common/Logger.h"
#include <algorithm>
#include <cmath>

namespace infer {

// Values per detection row: [x1, y1, x2, y2, confidence, class_id].
static constexpr int kRowStride  = 6;
// Default max detections emitted by the end-to-end head when the caller does
// not report the buffer size.
static constexpr int kDefaultDets = 300;

// output layout: [batch, num_dets, 6], row-major.
// Coords are absolute xyxy pixels; predictions are already de-duplicated, so
// no NMS is applied (nms_thresh is ignored).
std::vector<std::vector<Detection>> YOLO26Decoder::decode(
    const float*      output,
    int               batch_size,
    const InferShape& shape,
    float             conf_thresh,
    float             /*nms_thresh*/,
    size_t            output_count)
{
    int num_dets = kDefaultDets;
    if (output_count > 0 && batch_size > 0) {
        num_dets = static_cast<int>(
            output_count / static_cast<size_t>(batch_size) / kRowStride);
    }
    const float img_w = static_cast<float>(shape.width);
    const float img_h = static_cast<float>(shape.height);

    std::vector<std::vector<Detection>> results(batch_size);
    int total_dets = 0;
    for (int b = 0; b < batch_size; ++b) {
        const float* base =
            output + static_cast<size_t>(b) * num_dets * kRowStride;
        for (int d = 0; d < num_dets; ++d) {
            const float* row = base + static_cast<size_t>(d) * kRowStride;
            const float  conf = row[4];
            if (conf < conf_thresh) continue;  // filters padding + low scores

            Detection det;
            det.class_id   = static_cast<int>(std::lround(row[5]));
            det.confidence = conf;
            det.bbox       = {std::clamp(row[0], 0.f, img_w),
                              std::clamp(row[1], 0.f, img_h),
                              std::clamp(row[2], 0.f, img_w),
                              std::clamp(row[3], 0.f, img_h)};
            results[b].push_back(det);
        }
        total_dets += static_cast<int>(results[b].size());
    }
    LOG_DEBUG("YOLO26: batch_size={} dets={}", batch_size, total_dets);
    return results;
}

} // namespace infer
