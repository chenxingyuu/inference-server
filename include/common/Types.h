#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <chrono>
#include <optional>
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

    // Phase 7b: secondary classifier attributes injected by cascade pipeline.
    // Key: attribute name (e.g. "vehicle_type"), Value: label (e.g. "sedan").
    std::map<std::string, std::string> attributes;

    // Optional tracking id. Populated when stream tracker is enabled.
    std::optional<int64_t> track_id;
};

// ── Cascade configuration: secondary model triggered by primary detections ────
struct CascadeConfig {
    std::string      model_id;         // secondary model to trigger
    std::vector<int> trigger_classes;  // primary class ids that trigger secondary
    float            crop_expand{0.0f}; // bbox expansion ratio (e.g. 0.1 = +10%)
    std::string      attribute_key;    // key injected into Detection.attributes
};

// ── Per-stream metadata attached to each frame ────────────────────────────────
struct StreamMeta {
    std::string stream_id;
    std::string model_id;
    double      capture_ts{0.0};  // epoch seconds at capture
    // Monotonic timestamp captured alongside capture_ts.
    // Used for duration math (latency) without being affected by NTP/clock jumps.
    // Unit: nanoseconds since an unspecified steady_clock epoch.
    uint64_t    capture_mono_ns{0};
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
    int     cuda_device_id{0};              // CUDA device that owns this buffer
};

#ifdef BUILD_ASCEND_BACKEND
// ── NPU buffer descriptor (DVPP output, YUV420SP / NV12 format) ──────────────
// Analogous to GpuBuffer for CUDA. All pointers are device (HBM) pointers.
// `frame_ref` holds a ref-counted handle that calls acldvppFree when destroyed,
// returning the YUV surface to the DVPP output pool.
struct AscendBuffer {
    void* yuv_device{nullptr};           // YUV420SP in NPU device (HBM) memory
    int   width{0};
    int   height{0};
    int   device_id{0};
    std::shared_ptr<void> frame_ref;     // keeps DVPP buffer alive; deleter: acldvppFree
};
#endif // BUILD_ASCEND_BACKEND

// ── A decoded frame ready for preprocessing ──────────────────────────────────
struct Frame {
    cv::Mat    image;           // CPU path: BGR uint8, original resolution
    GpuBuffer  gpu_buf;         // GPU path: NVDEC NV12 output
    bool       is_gpu{false};
#ifdef BUILD_ASCEND_BACKEND
    AscendBuffer ascend_buf;    // NPU path: DVPP YUV420SP output
    bool         is_ascend{false};
#endif
    StreamMeta meta;
};

// ── A batch of preprocessed frames ready for inference ───────────────────────
struct Batch {
    std::vector<cv::Mat>    frames;     // CPU path: preprocessed float CHW
    std::vector<GpuBuffer>  gpu_frames; // GPU path: NVDEC NV12 buffers
    std::vector<StreamMeta> metas;
    bool                    is_gpu{false};
    int                     cuda_device_id{0}; // device that decoded this batch

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
    double                   queue_latency_ms{0.0}; // capture → InferWorker dequeue
    double                   infer_ms{0.0};          // backend preprocess + forward pass
    double                   decode_ms{0.0};          // NMS + bbox decode
    std::string              model_id;
    std::vector<Detection>   detections;
    uint64_t                 frame_seq{0};  // detection index for cascade secondary routing
    // Internal-only monotonic capture timestamp (ns). Not serialized.
    uint64_t                 frame_mono_ns{0};
    std::string              frame_local_path;   // local archived frame path
    std::string              frame_url;          // optional MinIO URL
    std::string              frame_upload_state; // pending/uploaded/failed/disabled
};

// ── Device type ───────────────────────────────────────────────────────────────
enum class DeviceType { CPU, CUDA, Ascend, MPS };

// ── YOLO model version ────────────────────────────────────────────────────────
enum class YOLOVersion { v5, v8, v11, v26, Unknown };

} // namespace infer
