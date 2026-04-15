#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include "common/Types.h"

namespace infer {

struct ServerConfig {
    int stream_pool_threads{32};
    int max_streams{100};
    int management_port{8080};  // HTTP management server port
};

// Model type: detector outputs bounding boxes; classifier outputs class probabilities.
enum class ModelType { Detector, Classifier };

struct ModelConfig {
    std::string id;
    YOLOVersion version{YOLOVersion::v8};
    DeviceType  backend{DeviceType::CUDA};
    ModelType   model_type{ModelType::Detector};

    // TensorRT
    std::string engine_path;

    // Ascend: batch_size -> .om path
    std::map<int, std::string> om_paths;

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
    bool        use_hwdec{false};  // true → NVDEC hardware decode
};

struct KafkaConfig {
    std::string brokers{"kafka:9092"};
    std::string topic{"inference-results"};
    int         batch_size{100};
    int         linger_ms{5};
    std::string compression{"lz4"};
    int         queue_capacity{10000};
};

struct AppConfig {
    ServerConfig            server;
    std::vector<ModelConfig> models;
    std::vector<StreamConfig> streams;
    KafkaConfig             kafka;

    // Find by id helpers
    const ModelConfig* findModel(const std::string& id) const;
    const StreamConfig* findStream(const std::string& id) const;
};

// Parse config.yaml → AppConfig. Throws std::runtime_error on invalid config.
AppConfig loadConfig(const std::string& yaml_path);

// Convert string → YOLOVersion
YOLOVersion parseYOLOVersion(const std::string& s);

// Convert string → DeviceType
DeviceType parseDeviceType(const std::string& s);

} // namespace infer
