#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
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

std::string currentTimestampText(const std::string& stream_id, bool include_stream_id) {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto tt = clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    if (include_stream_id && !stream_id.empty()) {
        oss << stream_id << " ";
    }
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << "."
        << std::setw(3)
        << std::setfill('0')
        << millis.count();
    return oss.str();
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

void drawTimestamp(cv::Mat& frame,
                   const std::string& stream_id,
                   bool include_stream_id,
                   int pos_x,
                   int pos_y,
                   float font_scale,
                   int font_thickness) {
    if (frame.empty()) return;

    const std::string label = currentTimestampText(stream_id, include_stream_id);
    const float scale = std::max(0.3f, font_scale);
    const int thickness = std::max(1, font_thickness);

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);

    const int x = std::max(0, std::min(pos_x, std::max(0, frame.cols - text_size.width - 6)));
    const int y = std::max(text_size.height + baseline + 2, std::min(pos_y, std::max(0, frame.rows - 2)));
    const int bg_h = text_size.height + baseline + 4;
    const int bg_w = std::min(frame.cols - x, text_size.width + 6);
    if (bg_w <= 0 || bg_h <= 0) return;

    cv::rectangle(frame, cv::Rect(x, y - text_size.height - baseline - 2, bg_w, bg_h), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x + 3, y - 2), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), thickness);
}

} // namespace overlay
} // namespace infer
