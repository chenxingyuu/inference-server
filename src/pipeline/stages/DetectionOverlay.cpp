#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace infer {

void mapDetectionsFromModelToFrame(
    std::vector<Detection>& detections,
    int frame_w,
    int frame_h,
    const InferShape& model_shape) {
    if (detections.empty()) return;
    if (frame_w <= 0 || frame_h <= 0) return;
    if (model_shape.width <= 0 || model_shape.height <= 0) return;
    const float sx = static_cast<float>(frame_w) / static_cast<float>(model_shape.width);
    const float sy = static_cast<float>(frame_h) / static_cast<float>(model_shape.height);
    if (sx == 1.f && sy == 1.f) return;
    for (auto& d : detections) {
        d.bbox.x1 = std::max(0.0f, d.bbox.x1 * sx);
        d.bbox.x2 = std::max(0.0f, d.bbox.x2 * sx);
        d.bbox.y1 = std::max(0.0f, d.bbox.y1 * sy);
        d.bbox.y2 = std::max(0.0f, d.bbox.y2 * sy);
    }
}

namespace overlay {
namespace {

cv::Rect clampRect(const BBox& box, int width, int height) {
    const int x1 = std::max(0, std::min(static_cast<int>(std::floor(box.x1)), width - 1));
    const int y1 = std::max(0, std::min(static_cast<int>(std::floor(box.y1)), height - 1));
    const int x2 = std::max(0, std::min(static_cast<int>(std::ceil(box.x2)), width));
    const int y2 = std::max(0, std::min(static_cast<int>(std::ceil(box.y2)), height));
    const int w = std::max(0, x2 - x1);
    const int h = std::max(0, y2 - y1);
    return cv::Rect(x1, y1, w, h);
}

std::string detectionLabel(const Detection& det) {
    std::string label = det.class_name.empty() ? ("cls_" + std::to_string(det.class_id)) : det.class_name;
    label += " " + std::to_string(static_cast<int>(det.confidence * 100.0f)) + "%";
    if (det.track_id.has_value()) {
        label += " #" + std::to_string(*det.track_id);
    }
    return label;
}

} // namespace

void drawDetections(cv::Mat& frame, const std::optional<InferResult>& result, float draw_conf_thresh, int line_thickness) {
    if (!result.has_value()) return;
    for (const auto& det : result->detections) {
        if (det.confidence < draw_conf_thresh) continue;
        cv::Rect rect = clampRect(det.bbox, frame.cols, frame.rows);
        if (rect.width <= 0 || rect.height <= 0) continue;

        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), std::max(1, line_thickness));
        const std::string label = detectionLabel(det);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int text_y = std::max(0, rect.y - text_size.height - baseline - 4);
        const cv::Rect bg(rect.x, text_y, std::min(frame.cols - rect.x, text_size.width + 6), text_size.height + baseline + 4);
        cv::rectangle(frame, bg, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(rect.x + 3, text_y + text_size.height + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

} // namespace overlay
} // namespace infer
