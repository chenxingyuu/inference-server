#include "common/Config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace infer {

YOLOVersion parseYOLOVersion(const std::string& s) {
    if (s == "yolov5" || s == "v5") return YOLOVersion::v5;
    if (s == "yolov8" || s == "v8") return YOLOVersion::v8;
    if (s == "yolo11" || s == "v11") return YOLOVersion::v11;
    if (s == "yolo26" || s == "v26") return YOLOVersion::v26;
    throw std::runtime_error("Unknown YOLO version: " + s);
}

DeviceType parseDeviceType(const std::string& s) {
    if (s == "tensorrt" || s == "cuda" || s == "trt") return DeviceType::CUDA;
    if (s == "ascend" || s == "acl")                  return DeviceType::Ascend;
    if (s == "cpu")                                    return DeviceType::CPU;
    throw std::runtime_error("Unknown device type: " + s);
}

const ModelConfig* AppConfig::findModel(const std::string& id) const {
    for (const auto& m : models)
        if (m.id == id) return &m;
    return nullptr;
}

const StreamConfig* AppConfig::findStream(const std::string& id) const {
    for (const auto& s : streams)
        if (s.id == id) return &s;
    return nullptr;
}

AppConfig loadConfig(const std::string& yaml_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("loadConfig: " + std::string(e.what()));
    }

    AppConfig cfg;

    // ── server ────────────────────────────────────────────────────────────────
    if (auto sn = root["server"]) {
        cfg.server.stream_pool_threads = sn["stream_pool_threads"].as<int>(32);
        cfg.server.max_streams         = sn["max_streams"].as<int>(100);
    }

    // ── models ────────────────────────────────────────────────────────────────
    for (const auto& mn : root["models"]) {
        ModelConfig m;
        m.id          = mn["id"].as<std::string>();
        m.version     = parseYOLOVersion(mn["version"].as<std::string>("yolov8"));
        m.backend     = parseDeviceType(mn["backend"].as<std::string>("tensorrt"));
        m.engine_path = mn["engine_path"].as<std::string>("");
        m.batch_size  = mn["batch_size"].as<int>(16);
        m.conf_thresh = mn["conf_thresh"].as<float>(0.4f);
        m.nms_thresh  = mn["nms_thresh"].as<float>(0.45f);
        m.device_id   = mn["device_id"].as<int>(0);
        m.num_classes = mn["num_classes"].as<int>(80);

        auto& is       = m.input_shape;
        is.height      = mn["input_size"][0].as<int>(640);
        is.width       = mn["input_size"][1].as<int>(640);
        is.channels    = 3;
        is.batch       = m.batch_size;

        if (auto op = mn["om_paths"]) {
            for (const auto& kv : op) {
                int bs = kv.first.as<int>();
                m.om_paths[bs] = kv.second.as<std::string>();
            }
        }

        if (auto cn = mn["class_names"]) {
            for (const auto& n : cn)
                m.class_names.push_back(n.as<std::string>());
        }

        cfg.models.push_back(std::move(m));
    }

    // ── streams ───────────────────────────────────────────────────────────────
    for (const auto& sn : root["streams"]) {
        StreamConfig s;
        s.id                = sn["id"].as<std::string>();
        s.url               = sn["url"].as<std::string>();
        s.model_id          = sn["model_id"].as<std::string>();
        s.sample_fps        = sn["sample_fps"].as<int>(5);
        s.reconnect_delay_ms = sn["reconnect_delay_ms"].as<int>(3000);
        cfg.streams.push_back(std::move(s));
    }

    // ── kafka ─────────────────────────────────────────────────────────────────
    if (auto kn = root["kafka"]) {
        cfg.kafka.brokers        = kn["brokers"].as<std::string>("kafka:9092");
        cfg.kafka.topic          = kn["topic"].as<std::string>("inference-results");
        cfg.kafka.batch_size     = kn["batch_size"].as<int>(100);
        cfg.kafka.linger_ms      = kn["linger_ms"].as<int>(5);
        cfg.kafka.compression    = kn["compression"].as<std::string>("lz4");
        cfg.kafka.queue_capacity = kn["queue_capacity"].as<int>(10000);
    }

    return cfg;
}

} // namespace infer
