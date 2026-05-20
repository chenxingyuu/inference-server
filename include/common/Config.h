#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <optional>
#include "common/Types.h"

namespace infer {

struct ServerConfig {
    int stream_pool_threads{32};
    int max_streams{100};
    std::string socket_path{"/var/run/infer.sock"};
    std::string ffmpeg_log_level{"warning"};  // FFmpeg log threshold (quiet/fatal/error/warning/info/debug/trace)
    // libavcodec AVCodecContext::thread_count for CPU software decode (ignored for NVDEC hwdec).
    // 0 = leave unset (libavcodec auto); 1 = single-threaded decode; >1 = frame/slice workers.
    int         ffmpeg_decode_threads{2};
    // sink.stream ffmpeg_x264 child process: libx264 -threads N. 0 = ffmpeg auto (all logical CPUs).
    int         ffmpeg_encode_threads{0};
    std::string log_level{"info"};            // spdlog level (trace/debug/info/warn/error/critical/off)
    // Triton-style model repository root (<root>/<model_id>/config.yaml + version subdirs). Empty disables scan.
    // If set, non-empty env INFER_MODEL_REPOSITORY overrides this value after YAML parse.
    std::string model_repository;
};

// Model type: detector outputs bounding boxes; classifier outputs class probabilities.
enum class ModelType { Detector, Classifier };
enum class TrackerType { None, ByteTrack, DeepSort };
enum class SamplingMode { FrameCount, TimeBased };

struct ByteTrackConfig {
    float high_det_thresh{0.5f};
    float low_det_thresh{0.1f};
    float match_iou_thresh{0.3f};
    int   min_hits_to_confirm{2};
    int   max_lost_frames{30};
};

struct ModelConfig {
    std::string id;
    YOLOVersion version{YOLOVersion::v8};
    DeviceType  backend{DeviceType::CUDA};
    ModelType   model_type{ModelType::Detector};

    // TensorRT
    std::string engine_path;

    // Ascend: batch_size -> .om path
    std::map<int, std::string> om_paths;

    // ONNX Runtime (CPU / MPS)
    std::string onnx_path;

    InferShape  input_shape{};
    int         batch_size{16};
    float       conf_thresh{0.4f};
    float       nms_thresh{0.45f};
    int         device_id{0};
    int         num_classes{80};
    std::vector<std::string> class_names;

    // Phase 7a: performance
    int buffer_pool_size{4};               // pre-allocated GPU buffer slots
    int instance_count{1};                 // parallel InferWorker instances
    std::vector<int> device_ids;           // per-instance device binding (falls back to device_id)

    // Phase 7b: cascade pipeline (DeepStream primary/secondary GIE style)
    std::vector<CascadeConfig> cascade;   // secondary models triggered by this model

    // Phase 7b: dynamic management (Triton-style)
    // preferred_batch_sizes: scheduler tries to fill batches to these sizes in order.
    // If empty, falls back to batch_size (legacy behaviour).
    std::vector<int> preferred_batch_sizes;
    int max_queue_delay_us{10000};         // max wait before flushing partial batch (µs)
};

enum class SourceType { RTSP, File };

struct StreamConfig {
    std::string id;
    std::string url;
    std::string model_id;
    int          sample_fps{5};
    SamplingMode sampling_mode{SamplingMode::FrameCount};
    int          reconnect_delay_ms{3000};
    int         max_reconnect_delay_ms{60000};  // Phase 10: exponential backoff ceiling
    int         degraded_threshold{5};           // consecutive failures before DEGRADED
    int         max_reconnect_attempts{5};       // consecutive failures before FAILED (terminal)
    int         open_timeout_ms{10000};          // avformat_open_input timeout
    int         read_timeout_ms{5000};           // av_read_frame timeout
    bool        use_hwdec{false};  // true → NVDEC hardware decode
    int         cuda_device_id{0}; // set programmatically by GpuDeviceAllocator; not from YAML
    bool        use_ascend_dvpp{false};  // Ascend: true → DVPP NPU hardware decode
    int         ascend_device_id{0};     // NPU device index for DVPP decoding
    int         ascend_vpc_out_width{0};  // 0 = no VPC; else stretch decode to this size
    int         ascend_vpc_out_height{0};
    TrackerType tracker{TrackerType::None};
    ByteTrackConfig byte_track{};
    SourceType  source_type{SourceType::RTSP};  // RTSP stream or local file
    bool        loop_file{false};               // re-open file on EOF (file sources only)
};

enum class EdgeDropPolicy { Block, DropOldest, DropNewest };

struct PipelineSourceConfig {
    std::string id;
    std::string url;
    int         reconnect_delay_ms{3000};
    int         max_reconnect_delay_ms{60000};
    int         degraded_threshold{5};
    int         max_reconnect_attempts{5};
    int         open_timeout_ms{10000};   // avformat_open_input timeout
    int         read_timeout_ms{5000};    // av_read_frame timeout
};

struct StageConfig {
    std::string id;
    std::string type;
    std::map<std::string, std::string> with;
};

struct EdgeConfig {
    std::string     from;
    std::string     to;
    int             capacity{256};
    EdgeDropPolicy  drop_policy{EdgeDropPolicy::Block};
};

struct PipelineConfig {
    std::string id;
    std::vector<StageConfig> nodes;
    std::vector<EdgeConfig>  edges;
};

struct TaskConfig {
    std::string id;
    std::string source_id;
    std::string pipeline_id;
    int          sample_fps{5};
    SamplingMode sampling_mode{SamplingMode::FrameCount};
    bool         use_hwdec{false};
    bool         use_ascend_dvpp{false};  // Ascend: task-scoped DVPP decode switch
    int          ascend_device_id{0};     // Ascend NPU device index for DVPP
};

struct KafkaConfig {
    std::string brokers{"kafka:9092"};
    std::string topic{"inference-results"};
    int         batch_size{100};
    int         linger_ms{5};
    std::string compression{"lz4"};
    int         queue_capacity{10000};
};

struct GrpcConfig {
    int port{50051};
    int max_connections{100};
};

struct RedisConfig {
    std::string host{"localhost"};
    int         port{6379};
    std::string stream_prefix{"inference"};
    int         max_len{1000};
    int         queue_capacity{10000};
};

// A named, typed publisher entry. type is one of: "kafka" | "grpc" | "redis".
struct PublisherConfig {
    std::string id;
    std::string type;
    KafkaConfig kafka;
    GrpcConfig  grpc;
    RedisConfig redis;
};

struct MinioConfig {
    bool        enabled{false};
    std::string endpoint;
    std::string bucket;
    std::string access_key;
    std::string secret_key;
    std::string region{"us-east-1"};
    bool        use_ssl{false};
    int         connect_timeout_ms{1500};
    int         request_timeout_ms{3000};
    int         max_retries{2};
};

struct FrameArchiveConfig {
    bool        enabled{false};
    bool        allow_gpu_frames{true};
    int         worker_count{1};
    std::string local_dir{"./data/frames"};
    int         save_interval{1};      // save every N frames
    int         jpeg_quality{90};      // [1,100]
    int         queue_capacity{4096};  // async archive queue
    MinioConfig minio;
};

struct AppConfig {
    ServerConfig            server;
    std::vector<ModelConfig> models;
    std::vector<StreamConfig> streams;
    std::vector<PipelineSourceConfig> sources;
    std::vector<PipelineConfig> pipelines;
    std::vector<TaskConfig>    tasks;
    std::vector<PublisherConfig> publishers;
    FrameArchiveConfig      frame_archive;

    // Find by id helpers
    const ModelConfig* findModel(const std::string& id) const;
    const StreamConfig* findStream(const std::string& id) const;
    const PipelineSourceConfig* findSource(const std::string& id) const;
    const PipelineConfig* findPipeline(const std::string& id) const;
    const TaskConfig* findTask(const std::string& id) const;
};

// Scan `<root>/<model_id>/config.yaml` and numeric `<root>/<model_id>/<version>/` dirs; merge at load time only.
// Optional per-model YAML keys: active_version (int), weight_file (string, file inside selected version dir).
std::vector<ModelConfig> scanModelRepository(const std::string& root);

// Parse config.yaml → AppConfig. Throws std::runtime_error on invalid config.
AppConfig loadConfig(const std::string& yaml_path);

// Convert string → YOLOVersion
YOLOVersion parseYOLOVersion(const std::string& s);

// Convert string → DeviceType
DeviceType parseDeviceType(const std::string& s);

// Convert DeviceType → canonical string (matches parseDeviceType round-trip)
std::string deviceTypeToStr(DeviceType d);

// Convert string → TrackerType
TrackerType parseTrackerType(const std::string& s);

// Convert string → SamplingMode ("frame_count" | "time_based")
SamplingMode parseSamplingMode(const std::string& s);
EdgeDropPolicy parseEdgeDropPolicy(const std::string& s);
// Validate stream-scoped ByteTrack config. Throws std::runtime_error on invalid values.
void validateByteTrackConfig(const ByteTrackConfig& cfg);

} // namespace infer
