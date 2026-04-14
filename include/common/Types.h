#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <opencv2/core/mat.hpp>

namespace infer {

// ── Bounding box ──────────────────────────────────────────────────────────────
struct BBox {
    float x1{0.f};
    float y1{0.f};
    float x2{0.f};
    float y2{0.f};

    float width()  const noexcept { return x2 - x1; }
    float height() const noexcept { return y2 - y1; }
    float area()   const noexcept { return width() * height(); }
};

// ── Single detection result ───────────────────────────────────────────────────
struct Detection {
    int         class_id{0};
    std::string class_name;
    float       confidence{0.f};
    BBox        bbox{};
};

// ── Per-stream metadata attached to each frame ────────────────────────────────
struct StreamMeta {
    std::string stream_id;
    std::string model_id;
    double      capture_ts{0.0};  // epoch seconds at capture
    uint64_t    frame_seq{0};
    int         orig_width{0};
    int         orig_height{0};
};

// ── A decoded frame ready for preprocessing ──────────────────────────────────
struct Frame {
    cv::Mat    image;        // BGR, uint8, original resolution
    StreamMeta meta;
};

// ── A batch of preprocessed frames ready for inference ───────────────────────
struct Batch {
    std::vector<cv::Mat>    frames;   // preprocessed (resized, normalized float)
    std::vector<StreamMeta> metas;

    int size() const noexcept { return static_cast<int>(frames.size()); }
    bool empty() const noexcept { return frames.empty(); }
};

// ── Shape descriptor for model input/output ───────────────────────────────────
struct InferShape {
    int batch{1};
    int channels{3};
    int height{640};
    int width{640};
};

// ── Per-frame inference result before Kafka publish ──────────────────────────
struct InferResult {
    std::string              stream_id;
    double                   frame_ts{0.0};
    double                   infer_ts{0.0};
    double                   latency_ms{0.0};
    std::string              model_id;
    std::vector<Detection>   detections;
};

// ── Device type ───────────────────────────────────────────────────────────────
enum class DeviceType { CPU, CUDA, Ascend };

// ── YOLO model version ────────────────────────────────────────────────────────
enum class YOLOVersion { v5, v8, v11, v26, Unknown };

} // namespace infer
