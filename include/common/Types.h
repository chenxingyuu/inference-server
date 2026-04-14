#pragma once

#include <string>
#include <vector>
#include <memory>
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

// ── GPU buffer descriptor (NVDEC output, NV12 format) ─────────────────────────
// All pointers are device pointers. void* avoids pulling cuda_runtime.h into
// every translation unit; callers in CUDA TUs cast to the appropriate types.
//
// `frame_ref` holds a reference-counted opaque handle (AVFrame*) that keeps the
// NVDEC buffer alive until this GpuBuffer is destroyed. When the last copy of
// `frame_ref` goes out of scope it calls av_frame_unref, returning the NV12
// surface to FFmpeg's hardware frame pool.
struct GpuBuffer {
    void*   y_data{nullptr};    // Y plane  (device ptr, NV12)
    void*   uv_data{nullptr};   // UV interleaved plane (device ptr, NV12)
    int     width{0};
    int     height{0};
    void*   cuda_stream{nullptr};            // cudaStream_t cast to void*
    std::shared_ptr<void> frame_ref;         // keeps AVFrame alive
};

// ── A decoded frame ready for preprocessing ──────────────────────────────────
struct Frame {
    cv::Mat    image;           // CPU path: BGR uint8, original resolution
    GpuBuffer  gpu_buf;         // GPU path: NVDEC NV12 output
    bool       is_gpu{false};
    StreamMeta meta;
};

// ── A batch of preprocessed frames ready for inference ───────────────────────
struct Batch {
    std::vector<cv::Mat>    frames;     // CPU path: preprocessed float CHW
    std::vector<GpuBuffer>  gpu_frames; // GPU path: NVDEC NV12 buffers
    std::vector<StreamMeta> metas;
    bool                    is_gpu{false};

    int  size()  const noexcept { return static_cast<int>(metas.size()); }
    bool empty() const noexcept { return metas.empty(); }
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
