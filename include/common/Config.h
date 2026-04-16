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
    int management_port{8080};  // HTTP management server port
};

// Model type: detector outputs bounding boxes; classifier outputs class probabilities.
enum class ModelType { Detector, Classifier };
enum class TrackerType { None, ByteTrack, DeepSort };

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

struct StreamConfig {
    std::string id;
    std::string url;
    std::string model_id;
    int         sample_fps{5};
    int         reconnect_delay_ms{3000};
    int         max_reconnect_delay_ms{60000};  // Phase 10: exponential backoff ceiling
    int         max_reconnect_attempts{5};       // Phase 10: consecutive failures before DEGRADED
    bool        use_hwdec{false};  // true → NVDEC hardware decode
    TrackerType tracker{TrackerType::None};
    ByteTrackConfig byte_track{};
};

enum class EdgeDropPolicy { Block, DropOldest, DropNewest };

struct PipelineSourceConfig {
    std::string id;
    std::string url;
    int         sample_fps{5};
    int         reconnect_delay_ms{3000};
    int         max_reconnect_delay_ms{60000};
    int         max_reconnect_attempts{5};
    bool        use_hwdec{false};
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
    std::string source_id;
    std::vector<StageConfig> nodes;
    std::vector<EdgeConfig>  edges;
};

struct KafkaConfig {
    std::string brokers{"kafka:9092"};
    std::string topic{"inference-results"};
    int         batch_size{100};
    int         linger_ms{5};
    std::string compression{"lz4"};
    int         queue_capacity{10000};
    // Phase 10: heartbeat
    std::string heartbeat_topic{"inference-heartbeat"};
    int         heartbeat_interval_ms{5000};
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
    KafkaConfig             kafka;
    FrameArchiveConfig      frame_archive;

    // Find by id helpers
    const ModelConfig* findModel(const std::string& id) const;
    const StreamConfig* findStream(const std::string& id) const;
    const PipelineSourceConfig* findSource(const std::string& id) const;
    const PipelineConfig* findPipeline(const std::string& id) const;
};

// Parse config.yaml → AppConfig. Throws std::runtime_error on invalid config.
AppConfig loadConfig(const std::string& yaml_path);

// Convert string → YOLOVersion
YOLOVersion parseYOLOVersion(const std::string& s);

// Convert string → DeviceType
DeviceType parseDeviceType(const std::string& s);

// Convert string → TrackerType
TrackerType parseTrackerType(const std::string& s);
EdgeDropPolicy parseEdgeDropPolicy(const std::string& s);
// Validate stream-scoped ByteTrack config. Throws std::runtime_error on invalid values.
void validateByteTrackConfig(const ByteTrackConfig& cfg);

} // namespace infer
