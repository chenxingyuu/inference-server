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
    if (s == "mps" || s == "metal")                   return DeviceType::MPS;
    throw std::runtime_error("Unknown device type: " + s);
}

TrackerType parseTrackerType(const std::string& s) {
    if (s == "none") return TrackerType::None;
    if (s == "bytetrack") return TrackerType::ByteTrack;
    if (s == "deepsort") return TrackerType::DeepSort;
    throw std::runtime_error("Unknown tracker type: " + s);
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
        cfg.server.management_port     = sn["management_port"].as<int>(8080);
    }

    // ── models ────────────────────────────────────────────────────────────────
    for (const auto& mn : root["models"]) {
        ModelConfig m;
        m.id          = mn["id"].as<std::string>();
        m.version     = parseYOLOVersion(mn["version"].as<std::string>("yolov8"));
        m.backend     = parseDeviceType(mn["backend"].as<std::string>("tensorrt"));
        m.engine_path = mn["engine_path"].as<std::string>("");
        m.onnx_path   = mn["onnx_path"].as<std::string>("");
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

        // Phase 7b: cascade pipeline (primary → secondary)
        if (auto cas = mn["cascade"]) {
            for (const auto& cn : cas) {
                CascadeConfig cc;
                cc.model_id     = cn["model_id"].as<std::string>();
                cc.crop_expand  = cn["crop_expand"].as<float>(0.0f);
                cc.attribute_key = cn["attribute_key"].as<std::string>("");
                if (auto tc = cn["trigger_classes"]) {
                    for (const auto& c : tc)
                        cc.trigger_classes.push_back(c.as<int>());
                }
                m.cascade.push_back(std::move(cc));
            }
        }
        if (mn["model_type"].as<std::string>("detector") == "classifier")
            m.model_type = ModelType::Classifier;

        // Phase 7a: performance options
        m.buffer_pool_size = mn["buffer_pool_size"].as<int>(4);
        m.instance_count   = mn["instance_count"].as<int>(1);
        if (auto ids = mn["device_ids"]) {
            for (const auto& id : ids)
                m.device_ids.push_back(id.as<int>());
        }

        // Phase 7b: Triton-style batching
        m.max_queue_delay_us = mn["max_queue_delay_us"].as<int>(10000);
        if (auto pbs = mn["preferred_batch_sizes"]) {
            for (const auto& s : pbs)
                m.preferred_batch_sizes.push_back(s.as<int>());
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
        s.reconnect_delay_ms     = sn["reconnect_delay_ms"].as<int>(3000);
        s.max_reconnect_delay_ms = sn["max_reconnect_delay_ms"].as<int>(60000);
        s.max_reconnect_attempts = sn["max_reconnect_attempts"].as<int>(5);
        s.use_hwdec              = sn["use_hwdec"].as<bool>(false);
        s.tracker                = parseTrackerType(sn["tracker"].as<std::string>("none"));
        cfg.streams.push_back(std::move(s));
    }

    // ── kafka ─────────────────────────────────────────────────────────────────
    if (auto kn = root["kafka"]) {
        cfg.kafka.brokers               = kn["brokers"].as<std::string>("kafka:9092");
        cfg.kafka.topic                 = kn["topic"].as<std::string>("inference-results");
        cfg.kafka.batch_size            = kn["batch_size"].as<int>(100);
        cfg.kafka.linger_ms             = kn["linger_ms"].as<int>(5);
        cfg.kafka.compression           = kn["compression"].as<std::string>("lz4");
        cfg.kafka.queue_capacity        = kn["queue_capacity"].as<int>(10000);
        cfg.kafka.heartbeat_topic       = kn["heartbeat_topic"].as<std::string>("inference-heartbeat");
        cfg.kafka.heartbeat_interval_ms = kn["heartbeat_interval_ms"].as<int>(5000);
    }

    // ── frame_archive ──────────────────────────────────────────────────────────
    if (auto an = root["frame_archive"]) {
        cfg.frame_archive.enabled        = an["enabled"].as<bool>(false);
        cfg.frame_archive.local_dir      = an["local_dir"].as<std::string>("./data/frames");
        cfg.frame_archive.save_interval  = an["save_interval"].as<int>(1);
        cfg.frame_archive.jpeg_quality   = an["jpeg_quality"].as<int>(90);
        cfg.frame_archive.queue_capacity = an["queue_capacity"].as<int>(4096);

        if (cfg.frame_archive.save_interval <= 0) {
            throw std::runtime_error("frame_archive.save_interval must be >= 1");
        }
        if (cfg.frame_archive.jpeg_quality < 1 || cfg.frame_archive.jpeg_quality > 100) {
            throw std::runtime_error("frame_archive.jpeg_quality must be in [1, 100]");
        }
        if (cfg.frame_archive.queue_capacity <= 0) {
            throw std::runtime_error("frame_archive.queue_capacity must be >= 1");
        }

        if (auto mn = an["minio"]) {
            cfg.frame_archive.minio.enabled            = mn["enabled"].as<bool>(false);
            cfg.frame_archive.minio.endpoint           = mn["endpoint"].as<std::string>("");
            cfg.frame_archive.minio.bucket             = mn["bucket"].as<std::string>("");
            cfg.frame_archive.minio.access_key         = mn["access_key"].as<std::string>("");
            cfg.frame_archive.minio.secret_key         = mn["secret_key"].as<std::string>("");
            cfg.frame_archive.minio.region             = mn["region"].as<std::string>("us-east-1");
            cfg.frame_archive.minio.use_ssl            = mn["use_ssl"].as<bool>(false);
            cfg.frame_archive.minio.connect_timeout_ms = mn["connect_timeout_ms"].as<int>(1500);
            cfg.frame_archive.minio.request_timeout_ms = mn["request_timeout_ms"].as<int>(3000);
            cfg.frame_archive.minio.max_retries        = mn["max_retries"].as<int>(2);

            if (cfg.frame_archive.minio.enabled) {
                if (cfg.frame_archive.minio.endpoint.empty() || cfg.frame_archive.minio.bucket.empty()) {
                    throw std::runtime_error("frame_archive.minio endpoint/bucket required when enabled");
                }
                if (cfg.frame_archive.minio.access_key.empty() || cfg.frame_archive.minio.secret_key.empty()) {
                    throw std::runtime_error("frame_archive.minio access_key/secret_key required when enabled");
                }
            }
        }
    }

    return cfg;
}

} // namespace infer
