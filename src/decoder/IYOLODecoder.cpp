#include "decoder/IYOLODecoder.h"
#include <algorithm>
#include <numeric>

namespace infer {

float iou(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    if (inter == 0.f) return 0.f;
    return inter / (a.area() + b.area() - inter);
}

std::vector<int> nonMaxSuppression(
    const std::vector<Detection>& dets,
    float iou_thresh)
{
    // Sort by confidence descending
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return dets[a].confidence > dets[b].confidence;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int>  keep;

    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
        int a = idx[i];
        if (suppressed[a]) continue;
        keep.push_back(a);
        for (int j = i + 1; j < static_cast<int>(idx.size()); ++j) {
            int b = idx[j];
            if (!suppressed[b] && iou(dets[a].bbox, dets[b].bbox) > iou_thresh) {
                suppressed[b] = true;
            }
        }
    }
    return keep;
}

} // namespace infer
