#pragma once

#include "pipeline/Event.h"

#include <opencv2/core/mat.hpp>

#include <optional>
#include <vector>

namespace infer {

// Decoders emit boxes in model-input pixel space (same W×H as the resized tensor).
// Preprocess stretches the source frame to that size; map coordinates back to the
// decoded frame before drawing, tracking crops, or publishing pixel boxes.
void mapDetectionsFromModelToFrame(
    std::vector<Detection>& detections,
    int frame_w,
    int frame_h,
    const InferShape& model_shape);

namespace overlay {

void drawDetections(cv::Mat& frame, const std::optional<InferResult>& result, float draw_conf_thresh, int line_thickness);
void drawTimestamp(cv::Mat& frame,
                   const std::string& stream_id,
                   bool include_stream_id,
                   int pos_x,
                   int pos_y,
                   float font_scale,
                   int font_thickness);

} // namespace overlay
} // namespace infer
