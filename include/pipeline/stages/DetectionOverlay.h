#pragma once

#include "pipeline/Event.h"

#include <opencv2/core/mat.hpp>

#include <optional>

namespace infer {
namespace overlay {

void drawDetections(cv::Mat& frame, const std::optional<InferResult>& result, float draw_conf_thresh, int line_thickness);

} // namespace overlay
} // namespace infer
