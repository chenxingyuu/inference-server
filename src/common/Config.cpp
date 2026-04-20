#include "common/Config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace infer {

namespace {

std::string scalarToString(const YAML::Node& node) {
    if (!node.IsScalar()) {
        throw std::runtime_error("stage.with values must be scalar values");
    }
    return node.as<std::string>();
}

void validateSources(const AppConfig& cfg) {
    std::unordered_set<std::string> source_ids;
    for (const auto& src : cfg.sources) {
        if (src.id.empty()) throw std::runtime_error("source.id must not be empty");
        if (!source_ids.insert(src.id).second) {
            throw std::runtime_error("duplicate source id: " + src.id);
        }
    }
}

void validatePipelineGraphs(const AppConfig& cfg) {
    std::unordered_set<std::string> pipeline_ids;
    for (const auto& p : cfg.pipelines) {
        if (p.id.empty()) throw std::runtime_error("pipeline.id must not be empty");
        if (!pipeline_ids.insert(p.id).second) {
            throw std::runtime_error("duplicate pipeline id: " + p.id);
        }
        if (p.nodes.empty()) throw std::runtime_error("pipeline.nodes must not be empty: " + p.id);
        std::unordered_set<std::string> node_ids;
        std::unordered_map<std::string, int> indegree;
        std::unordered_map<std::string, std::vector<std::string>> graph;
        for (const auto& n : p.nodes) {
            if (n.id.empty()) throw std::runtime_error("pipeline node id must not be empty");
            if (!node_ids.insert(n.id).second) {
                throw std::runtime_error("duplicate pipeline node id: " + n.id);
            }
            if (n.type == "sink.stream") {
                auto output_url_it = n.with.find("output_url");
                if (output_url_it == n.with.end() || output_url_it->second.empty()) {
                    throw std::runtime_error("sink.stream requires with.output_url");
                }
                auto protocol_it = n.with.find("protocol");
                const std::string protocol = (protocol_it == n.with.end()) ? "rtsp" : protocol_it->second;
                if (protocol != "rtsp" && protocol != "rtmp") {
                    throw std::runtime_error("sink.stream protocol must be one of: rtsp, rtmp");
                }
            }
            if (n.type == "sink.ffplay") {
                auto fps_it = n.with.find("fps");
                if (fps_it != n.with.end()) {
                    float fps = 0.0f;
                    try {
                        fps = std::stof(fps_it->second);
                    } catch (const std::exception&) {
                        throw std::runtime_error("sink.ffplay invalid fps");
                    }
                    if (fps <= 0.0f) {
                        throw std::runtime_error("sink.ffplay fps must be > 0");
                    }
                }
                auto qc_it = n.with.find("queue_capacity");
                if (qc_it != n.with.end()) {
                    int q = 0;
                    try {
                        q = std::stoi(qc_it->second);
                    } catch (const std::exception&) {
                        throw std::runtime_error("sink.ffplay invalid queue_capacity");
                    }
                    if (q < 1) {
                        throw std::runtime_error("sink.ffplay queue_capacity must be >= 1");
                    }
                }
                auto dp_it = n.with.find("drop_policy");
                if (dp_it != n.with.end()) {
                    if (dp_it->second != "drop_oldest" && dp_it->second != "drop_newest") {
                        throw std::runtime_error("sink.ffplay drop_policy must be one of: drop_oldest, drop_newest");
                    }
                }
            }
            indegree[n.id] = 0;
        }
        for (const auto& e : p.edges) {
            if (!node_ids.count(e.from)) throw std::runtime_error("edge.from not found: " + e.from);
            if (!node_ids.count(e.to)) throw std::runtime_error("edge.to not found: " + e.to);
            if (e.capacity <= 0) throw std::runtime_error("edge.capacity must be >= 1");
            graph[e.from].push_back(e.to);
            indegree[e.to] += 1;
        }
        std::queue<std::string> q;
        for (const auto& [id, deg] : indegree) {
            if (deg == 0) q.push(id);
        }
        int visited = 0;
        while (!q.empty()) {
            auto cur = q.front();
            q.pop();
            visited++;
            for (const auto& nxt : graph[cur]) {
                auto it = indegree.find(nxt);
                if (--(it->second) == 0) q.push(nxt);
            }
        }
        if (visited != static_cast<int>(p.nodes.size())) {
            throw std::runtime_error("pipeline graph has cycle: " + p.id);
        }
    }
}

void validateTasks(const AppConfig& cfg) {
    if (cfg.tasks.empty()) {
        throw std::runtime_error("tasks must not be empty");
    }
    std::unordered_set<std::string> task_ids;
    for (const auto& t : cfg.tasks) {
        if (t.id.empty()) throw std::runtime_error("task.id must not be empty");
        if (!task_ids.insert(t.id).second) {
            throw std::runtime_error("duplicate task id: " + t.id);
        }
        if (t.source_id.empty()) throw std::runtime_error("task.source_id must not be empty: " + t.id);
        if (t.pipeline_id.empty()) throw std::runtime_error("task.pipeline_id must not be empty: " + t.id);
        if (!cfg.findSource(t.source_id)) {
            throw std::runtime_error("task source not found: " + t.source_id + " (task " + t.id + ")");
        }
        if (!cfg.findPipeline(t.pipeline_id)) {
            throw std::runtime_error("task pipeline not found: " + t.pipeline_id + " (task " + t.id + ")");
        }
        if (t.sample_fps < 1) {
            throw std::runtime_error("task.sample_fps must be >= 1 (task " + t.id + ")");
        }
    }
}

void validateAppConfig(const AppConfig& cfg) {
    validateSources(cfg);
    validatePipelineGraphs(cfg);
    validateTasks(cfg);
}

} // namespace

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

EdgeDropPolicy parseEdgeDropPolicy(const std::string& s) {
    if (s == "block") return EdgeDropPolicy::Block;
    if (s == "drop_oldest") return EdgeDropPolicy::DropOldest;
    if (s == "drop_newest") return EdgeDropPolicy::DropNewest;
    throw std::runtime_error("Unknown edge drop policy: " + s);
}

void validateByteTrackConfig(const ByteTrackConfig& cfg) {
    if (cfg.low_det_thresh < 0.0f || cfg.low_det_thresh > 1.0f) {
        throw std::runtime_error("bytetrack.low_det_thresh must be in [0, 1]");
    }
    if (cfg.high_det_thresh < 0.0f || cfg.high_det_thresh > 1.0f) {
        throw std::runtime_error("bytetrack.high_det_thresh must be in [0, 1]");
    }
    if (cfg.low_det_thresh > cfg.high_det_thresh) {
        throw std::runtime_error("bytetrack.low_det_thresh must be <= high_det_thresh");
    }
    if (cfg.match_iou_thresh < 0.0f || cfg.match_iou_thresh > 1.0f) {
        throw std::runtime_error("bytetrack.match_iou_thresh must be in [0, 1]");
    }
    if (cfg.min_hits_to_confirm < 1) {
        throw std::runtime_error("bytetrack.min_hits_to_confirm must be >= 1");
    }
    if (cfg.max_lost_frames < 1) {
        throw std::runtime_error("bytetrack.max_lost_frames must be >= 1");
    }
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

const PipelineSourceConfig* AppConfig::findSource(const std::string& id) const {
    for (const auto& s : sources)
        if (s.id == id) return &s;
    return nullptr;
}

const PipelineConfig* AppConfig::findPipeline(const std::string& id) const {
    for (const auto& p : pipelines)
        if (p.id == id) return &p;
    return nullptr;
}

const TaskConfig* AppConfig::findTask(const std::string& id) const {
    for (const auto& t : tasks)
        if (t.id == id) return &t;
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
    if (auto sn = root["server"]) {
        cfg.server.stream_pool_threads = sn["stream_pool_threads"].as<int>(32);
        cfg.server.max_streams         = sn["max_streams"].as<int>(100);
        cfg.server.management_port     = sn["management_port"].as<int>(8080);
    }
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
        auto& is      = m.input_shape;
        is.height     = mn["input_size"][0].as<int>(640);
        is.width      = mn["input_size"][1].as<int>(640);
        is.channels   = 3;
        is.batch      = m.batch_size;
        if (auto op = mn["om_paths"]) {
            for (const auto& kv : op) m.om_paths[kv.first.as<int>()] = kv.second.as<std::string>();
        }
        if (auto cn = mn["class_names"]) {
            for (const auto& n : cn) m.class_names.push_back(n.as<std::string>());
        }
        if (auto cas = mn["cascade"]) {
            for (const auto& cn : cas) {
                CascadeConfig cc;
                cc.model_id      = cn["model_id"].as<std::string>();
                cc.crop_expand   = cn["crop_expand"].as<float>(0.0f);
                cc.attribute_key = cn["attribute_key"].as<std::string>("");
                if (auto tc = cn["trigger_classes"]) {
                    for (const auto& c : tc) cc.trigger_classes.push_back(c.as<int>());
                }
                m.cascade.push_back(std::move(cc));
            }
        }
        if (mn["model_type"].as<std::string>("detector") == "classifier") {
            m.model_type = ModelType::Classifier;
        }
        m.buffer_pool_size = mn["buffer_pool_size"].as<int>(4);
        m.instance_count   = mn["instance_count"].as<int>(1);
        if (auto ids = mn["device_ids"]) {
            for (const auto& id : ids) m.device_ids.push_back(id.as<int>());
        }
        m.max_queue_delay_us = mn["max_queue_delay_us"].as<int>(10000);
        if (auto pbs = mn["preferred_batch_sizes"]) {
            for (const auto& s : pbs) m.preferred_batch_sizes.push_back(s.as<int>());
        }
        cfg.models.push_back(std::move(m));
    }
    for (const auto& sn : root["sources"]) {
        if (sn["sample_fps"]) {
            throw std::runtime_error(
                "sources[].sample_fps is no longer supported; set tasks[].sample_fps per task (source id=" +
                sn["id"].as<std::string>("") + ")");
        }
        if (sn["use_hwdec"]) {
            throw std::runtime_error(
                "sources[].use_hwdec is no longer supported; set tasks[].use_hwdec per task (source id=" +
                sn["id"].as<std::string>("") + ")");
        }
        PipelineSourceConfig s;
        s.id                     = sn["id"].as<std::string>();
        s.url                    = sn["url"].as<std::string>();
        s.reconnect_delay_ms     = sn["reconnect_delay_ms"].as<int>(3000);
        s.max_reconnect_delay_ms = sn["max_reconnect_delay_ms"].as<int>(60000);
        s.degraded_threshold     = sn["degraded_threshold"].as<int>(5);
        s.max_reconnect_attempts = sn["max_reconnect_attempts"].as<int>(5);
        cfg.sources.push_back(std::move(s));
    }
    for (const auto& pn : root["pipelines"]) {
        PipelineConfig p;
        p.id = pn["id"].as<std::string>();
        for (const auto& nn : pn["nodes"]) {
            StageConfig st;
            st.id = nn["id"].as<std::string>();
            st.type = nn["type"].as<std::string>();
            if (auto with = nn["with"]) {
                for (auto it = with.begin(); it != with.end(); ++it) {
                    st.with[it->first.as<std::string>()] = scalarToString(it->second);
                }
            }
            p.nodes.push_back(std::move(st));
        }
        for (const auto& en : pn["edges"]) {
            EdgeConfig e;
            e.from = en["from"].as<std::string>();
            e.to = en["to"].as<std::string>();
            if (auto q = en["queue"]) {
                e.capacity = q["capacity"].as<int>(256);
                e.drop_policy = parseEdgeDropPolicy(q["drop_policy"].as<std::string>("block"));
            }
            p.edges.push_back(std::move(e));
        }
        cfg.pipelines.push_back(std::move(p));
    }
    if (auto tasks_node = root["tasks"]) {
        for (const auto& tn : tasks_node) {
            TaskConfig t;
            t.id = tn["id"].as<std::string>();
            t.source_id = tn["source_id"].as<std::string>();
            t.pipeline_id = tn["pipeline_id"].as<std::string>();
            t.sample_fps = tn["sample_fps"].as<int>(5);
            t.use_hwdec = tn["use_hwdec"].as<bool>(false);
            cfg.tasks.push_back(std::move(t));
        }
    }
    if (auto kn = root["kafka"]) {
        cfg.kafka.brokers               = kn["brokers"].as<std::string>("kafka:9092");
        cfg.kafka.topic                 = kn["topic"].as<std::string>("inference-results");
        cfg.kafka.batch_size            = kn["batch_size"].as<int>(100);
        cfg.kafka.linger_ms             = kn["linger_ms"].as<int>(5);
        cfg.kafka.compression           = kn["compression"].as<std::string>("lz4");
        cfg.kafka.queue_capacity        = kn["queue_capacity"].as<int>(10000);
        cfg.kafka.heartbeat_topic       = kn["heartbeat_topic"].as<std::string>("inference-heartbeat");
        cfg.kafka.heartbeat_interval_ms = kn["heartbeat_interval_ms"].as<int>(5000);
        cfg.kafka.control_topic         = kn["control_topic"].as<std::string>("inference-control");
    }
    if (auto an = root["frame_archive"]) {
        cfg.frame_archive.enabled        = an["enabled"].as<bool>(false);
        cfg.frame_archive.local_dir      = an["local_dir"].as<std::string>("./data/frames");
        cfg.frame_archive.save_interval  = an["save_interval"].as<int>(1);
        cfg.frame_archive.jpeg_quality   = an["jpeg_quality"].as<int>(90);
        cfg.frame_archive.queue_capacity = an["queue_capacity"].as<int>(4096);
        if (cfg.frame_archive.save_interval <= 0) throw std::runtime_error("frame_archive.save_interval must be >= 1");
        if (cfg.frame_archive.jpeg_quality < 1 || cfg.frame_archive.jpeg_quality > 100) throw std::runtime_error("frame_archive.jpeg_quality must be in [1, 100]");
        if (cfg.frame_archive.queue_capacity <= 0) throw std::runtime_error("frame_archive.queue_capacity must be >= 1");
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
        }
    }
    validateAppConfig(cfg);
    return cfg;
}

} // namespace infer
