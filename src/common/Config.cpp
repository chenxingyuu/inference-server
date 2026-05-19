#include "common/Config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <regex>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace infer {

namespace {

std::string scalarToString(const YAML::Node& node) {
    if (!node.IsScalar()) {
        throw std::runtime_error("stage.with values must be scalar values");
    }
    return node.as<std::string>();
}

void validateSources(const AppConfig& cfg) {
    static const std::regex kStreamIdPattern("^[A-Za-z0-9_-]+$");
    std::unordered_set<std::string> source_ids;
    for (const auto& src : cfg.sources) {
        if (src.id.empty()) throw std::runtime_error("source.id must not be empty");
        if (!std::regex_match(src.id, kStreamIdPattern)) {
            throw std::runtime_error("source.id contains illegal characters: " + src.id);
        }
        if (!source_ids.insert(src.id).second) {
            throw std::runtime_error("duplicate source id: " + src.id);
        }
    }
}

const std::regex& modelIdPattern() {
    static const std::regex kModelIdPattern("^[A-Za-z0-9_-]+$");
    return kModelIdPattern;
}

void validateModels(const AppConfig& cfg) {
    std::unordered_set<std::string> model_ids;
    for (const auto& model : cfg.models) {
        if (model.id.empty()) throw std::runtime_error("model.id must not be empty");
        if (!std::regex_match(model.id, modelIdPattern())) {
            throw std::runtime_error("model.id contains illegal characters: " + model.id);
        }
        if (!model_ids.insert(model.id).second) {
            throw std::runtime_error("duplicate model id: " + model.id);
        }
    }
}

void validateCascadeModelRefs(const AppConfig& cfg) {
    for (const auto& model : cfg.models) {
        for (const auto& cc : model.cascade) {
            if (!cfg.findModel(cc.model_id)) {
                throw std::runtime_error("cascade references unknown model_id: " + cc.model_id + " (from model " + model.id + ")");
            }
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
                    if (!std::isfinite(fps) || fps > 240.0f) {
                        throw std::runtime_error("sink.ffplay fps must be finite and <= 240");
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
        if (t.ascend_device_id < 0) {
            throw std::runtime_error("task.ascend_device_id must be >= 0 (task " + t.id + ")");
        }
    }
}

void validateAppConfig(const AppConfig& cfg) {
    validateModels(cfg);
    validateCascadeModelRefs(cfg);
    validateSources(cfg);
    validatePipelineGraphs(cfg);
    validateTasks(cfg);
}

} // namespace

YOLOVersion parseYOLOVersion(const std::string& s);
DeviceType  parseDeviceType(const std::string& s);

namespace {

namespace fs = std::filesystem;

std::string pathGeneric(const fs::path& p) { return p.lexically_normal().generic_string(); }

void resolveRelativeWeightPaths(ModelConfig& m, const fs::path& version_dir) {
    auto resolve = [&](std::string& p) {
        if (p.empty()) return;
        fs::path fp(p);
        if (fp.is_absolute()) return;
        p = pathGeneric(version_dir / fp);
    };
    resolve(m.engine_path);
    resolve(m.onnx_path);
    for (auto& kv : m.om_paths) resolve(kv.second);
}

void collectExtensionMatches(const fs::path& dir, const char* ext, std::vector<fs::path>& out) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;
    for (const auto& ent : fs::directory_iterator(dir)) {
        if (!ent.is_regular_file()) continue;
        if (ent.path().extension() == ext) out.push_back(ent.path());
    }
}

void applyWeightFile(ModelConfig& m, const fs::path& version_dir, const std::string& weight_file) {
    if (weight_file.empty()) return;
    const fs::path wf = version_dir / weight_file;
    if (!fs::is_regular_file(wf)) {
        throw std::runtime_error("model_repository model " + m.id + ": weight_file not found: " + pathGeneric(wf));
    }
    const std::string ext = wf.extension().string();
    if (ext == ".onnx" || ext == ".ONNX") {
        if (m.backend != DeviceType::CPU && m.backend != DeviceType::MPS) {
            throw std::runtime_error("model_repository model " + m.id + ": weight_file .onnx requires backend cpu or mps");
        }
        m.onnx_path = pathGeneric(wf);
        return;
    }
    if (ext == ".engine" || ext == ".ENGINE") {
        if (m.backend != DeviceType::CUDA) {
            throw std::runtime_error("model_repository model " + m.id + ": weight_file .engine requires tensorrt/cuda backend");
        }
        m.engine_path = pathGeneric(wf);
        return;
    }
    if (ext == ".om" || ext == ".OM") {
        if (m.backend != DeviceType::Ascend) {
            throw std::runtime_error("model_repository model " + m.id + ": weight_file .om requires ascend backend");
        }
        m.om_paths[m.batch_size] = pathGeneric(wf);
        return;
    }
    throw std::runtime_error("model_repository model " + m.id + ": weight_file has unsupported extension: " + ext);
}

void autoDetectWeights(ModelConfig& m, const fs::path& version_dir, const std::string& weight_file) {
    applyWeightFile(m, version_dir, weight_file);
    if (!weight_file.empty()) return;

    if (m.backend == DeviceType::CUDA) {
        if (!m.engine_path.empty() || !m.onnx_path.empty()) return;
        std::vector<fs::path> engines;
        collectExtensionMatches(version_dir, ".engine", engines);
        if (engines.empty()) {
            throw std::runtime_error("model_repository model " + m.id + ": no TensorRT .engine in version dir " + pathGeneric(version_dir));
        }
        if (engines.size() > 1) {
            throw std::runtime_error("model_repository model " + m.id + ": multiple .engine files in " + pathGeneric(version_dir)
                                     + "; set engine_path in config.yaml or use weight_file");
        }
        m.engine_path = pathGeneric(engines[0]);
        return;
    }
    if (m.backend == DeviceType::CPU || m.backend == DeviceType::MPS) {
        if (!m.onnx_path.empty()) return;
        std::vector<fs::path> onnxes;
        collectExtensionMatches(version_dir, ".onnx", onnxes);
        if (onnxes.empty()) {
            throw std::runtime_error("model_repository model " + m.id + ": no .onnx in version dir " + pathGeneric(version_dir));
        }
        if (onnxes.size() > 1) {
            throw std::runtime_error("model_repository model " + m.id + ": multiple .onnx files in " + pathGeneric(version_dir)
                                     + "; set onnx_path in config.yaml or use weight_file");
        }
        m.onnx_path = pathGeneric(onnxes[0]);
        return;
    }
    if (m.backend == DeviceType::Ascend) {
        if (!m.om_paths.empty()) return;
        std::vector<fs::path> oms;
        collectExtensionMatches(version_dir, ".om", oms);
        if (oms.empty()) {
            throw std::runtime_error("model_repository model " + m.id + ": no om_paths in config and no .om in " + pathGeneric(version_dir));
        }
        if (oms.size() > 1) {
            throw std::runtime_error("model_repository model " + m.id + ": multiple .om files in " + pathGeneric(version_dir)
                                     + "; set om_paths in config.yaml or use weight_file");
        }
        m.om_paths[m.batch_size] = pathGeneric(oms[0]);
        return;
    }
    throw std::runtime_error("model_repository model " + m.id + ": unsupported backend for auto weight detection");
}

void validateRepoModelWeights(const ModelConfig& m) {
    if (m.backend == DeviceType::CUDA) {
        if (m.engine_path.empty()) throw std::runtime_error("model_repository model " + m.id + ": engine_path is empty after scan");
        return;
    }
    if (m.backend == DeviceType::CPU || m.backend == DeviceType::MPS) {
        if (m.onnx_path.empty()) throw std::runtime_error("model_repository model " + m.id + ": onnx_path is empty after scan");
        return;
    }
    if (m.backend == DeviceType::Ascend) {
        if (m.om_paths.empty()) throw std::runtime_error("model_repository model " + m.id + ": om_paths is empty after scan");
        return;
    }
    throw std::runtime_error("model_repository model " + m.id + ": unsupported backend");
}

} // namespace

ModelConfig parseModelConfigNode(const YAML::Node& mn) {
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
        for (const auto& kv : op) {
            try {
                m.om_paths[kv.first.as<int>()] = kv.second.as<std::string>();
            } catch (const YAML::Exception& e) {
                throw std::runtime_error("om_paths keys must be integers: " + std::string(e.what()));
            }
        }
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
        if (static_cast<int>(m.device_ids.size()) != m.instance_count) {
            throw std::runtime_error(
                "model '" + m.id + "': device_ids has " +
                std::to_string(m.device_ids.size()) + " entries but instance_count=" +
                std::to_string(m.instance_count) +
                "; sizes must match");
        }
    }
    m.max_queue_delay_us = mn["max_queue_delay_us"].as<int>(10000);
    if (auto pbs = mn["preferred_batch_sizes"]) {
        for (const auto& s : pbs) m.preferred_batch_sizes.push_back(s.as<int>());
    }
    return m;
}

std::vector<ModelConfig> scanModelRepository(const std::string& root) {
    if (root.empty()) return {};
    const std::filesystem::path repo_root(root);
    if (!std::filesystem::exists(repo_root) || !std::filesystem::is_directory(repo_root)) {
        throw std::runtime_error("model_repository is not a directory: " + root);
    }

    std::vector<std::filesystem::path> model_dirs;
    for (const auto& ent : std::filesystem::directory_iterator(repo_root)) {
        if (!ent.is_directory()) continue;
        const std::string name = ent.path().filename().string();
        if (name.empty() || name.front() == '.') continue;
        if (!std::regex_match(name, modelIdPattern())) continue;
        const std::filesystem::path cfg_path = ent.path() / "config.yaml";
        if (!std::filesystem::is_regular_file(cfg_path)) continue;
        model_dirs.push_back(ent.path());
    }
    std::sort(model_dirs.begin(), model_dirs.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
        return a.filename().string() < b.filename().string();
    });

    std::vector<ModelConfig> out;
    out.reserve(model_dirs.size());
    for (const std::filesystem::path& model_dir : model_dirs) {
        const std::string model_id = model_dir.filename().string();
        YAML::Node mn;
        try {
            mn = YAML::LoadFile((model_dir / "config.yaml").generic_string());
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("model_repository model " + model_id + ": " + std::string(e.what()));
        }
        if (!mn.IsMap()) {
            throw std::runtime_error("model_repository model " + model_id + ": config.yaml must be a mapping");
        }
        if (mn["id"]) {
            const std::string yaml_id = mn["id"].as<std::string>();
            if (!yaml_id.empty() && yaml_id != model_id) {
                throw std::runtime_error("model_repository model " + model_id + ": id in config.yaml must match directory name (got id=" + yaml_id + ")");
            }
        }
        if (!mn["id"] || !mn["id"].IsDefined() || mn["id"].as<std::string>("").empty()) {
            mn["id"] = YAML::Node(model_id);
        }

        int         active_version = -1;
        std::string weight_file;
        if (mn["active_version"]) active_version = mn["active_version"].as<int>();
        if (mn["weight_file"]) weight_file = mn["weight_file"].as<std::string>("");

        std::vector<unsigned long> versions;
        for (const auto& ent : std::filesystem::directory_iterator(model_dir)) {
            if (!ent.is_directory()) continue;
            const std::string vname = ent.path().filename().string();
            if (vname.empty() || vname.front() == '.') continue;
            bool all_digit = !vname.empty();
            for (char c : vname) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    all_digit = false;
                    break;
                }
            }
            if (!all_digit) continue;
            try {
                const unsigned long v = std::stoul(vname);
                if (v == 0) continue;
                versions.push_back(v);
            } catch (const std::exception&) {
                continue;
            }
        }
        if (versions.empty()) {
            throw std::runtime_error("model_repository model " + model_id + ": no numeric version subdirectories");
        }
        std::sort(versions.begin(), versions.end());
        versions.erase(std::unique(versions.begin(), versions.end()), versions.end());

        unsigned long chosen = 0;
        if (active_version >= 0) {
            const auto want = static_cast<unsigned long>(active_version);
            if (std::find(versions.begin(), versions.end(), want) == versions.end()) {
                throw std::runtime_error("model_repository model " + model_id + ": active_version " + std::to_string(active_version)
                                         + " not found among version directories");
            }
            chosen = want;
        } else {
            chosen = *std::max_element(versions.begin(), versions.end());
        }

        const std::filesystem::path version_dir = model_dir / std::to_string(chosen);
        if (!std::filesystem::is_directory(version_dir)) {
            throw std::runtime_error("model_repository model " + model_id + ": missing version directory " + std::to_string(chosen));
        }

        ModelConfig m = parseModelConfigNode(mn);
        if (m.id != model_id) {
            throw std::runtime_error("model_repository model " + model_id + ": internal id mismatch");
        }
        resolveRelativeWeightPaths(m, version_dir);
        autoDetectWeights(m, version_dir, weight_file);
        validateRepoModelWeights(m);
        out.push_back(std::move(m));
    }
    return out;
}

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

std::string deviceTypeToStr(DeviceType d) {
    switch (d) {
        case DeviceType::CUDA:   return "tensorrt";
        case DeviceType::Ascend: return "ascend";
        case DeviceType::CPU:    return "cpu";
        case DeviceType::MPS:    return "mps";
    }
    return "cpu";
}

TrackerType parseTrackerType(const std::string& s) {
    if (s == "none") return TrackerType::None;
    if (s == "bytetrack") return TrackerType::ByteTrack;
    if (s == "deepsort") return TrackerType::DeepSort;
    throw std::runtime_error("Unknown tracker type: " + s);
}

SamplingMode parseSamplingMode(const std::string& s) {
    if (s == "frame_count" || s.empty()) return SamplingMode::FrameCount;
    if (s == "time_based")               return SamplingMode::TimeBased;
    throw std::runtime_error("Unknown sampling_mode: " + s);
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
        cfg.server.socket_path         = sn["socket_path"].as<std::string>("/var/run/infer.sock");
        cfg.server.ffmpeg_log_level    = sn["ffmpeg_log_level"].as<std::string>("warning");
        cfg.server.log_level           = sn["log_level"].as<std::string>("info");
        cfg.server.model_repository    = sn["model_repository"].as<std::string>("");
    }
    if (root["models"]) {
        if (!root["models"].IsSequence()) {
            throw std::runtime_error("models must be a YAML sequence");
        }
        for (const auto& mn : root["models"]) {
            cfg.models.push_back(parseModelConfigNode(mn));
        }
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
        s.open_timeout_ms        = sn["open_timeout_ms"].as<int>(10000);
        s.read_timeout_ms        = sn["read_timeout_ms"].as<int>(5000);
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
            t.sample_fps     = tn["sample_fps"].as<int>(5);
            t.sampling_mode  = parseSamplingMode(tn["sampling_mode"].as<std::string>("frame_count"));
            t.use_hwdec      = tn["use_hwdec"].as<bool>(false);
            t.use_ascend_dvpp = tn["use_ascend_dvpp"].as<bool>(false);
            t.ascend_device_id = tn["ascend_device_id"].as<int>(0);
            cfg.tasks.push_back(std::move(t));
        }
    }
    auto parseKafkaNode = [](const YAML::Node& kn, KafkaConfig& k) {
        k.enabled               = kn["enabled"].as<bool>(true);
        k.brokers               = kn["brokers"].as<std::string>("kafka:9092");
        k.topic                 = kn["topic"].as<std::string>("inference-results");
        k.batch_size            = kn["batch_size"].as<int>(100);
        k.linger_ms             = kn["linger_ms"].as<int>(5);
        k.compression           = kn["compression"].as<std::string>("lz4");
        k.queue_capacity        = kn["queue_capacity"].as<int>(10000);
    };
    if (auto pn = root["publishers"]) {
        if (auto kn = pn["kafka"]) parseKafkaNode(kn, cfg.publishers.kafka);
        if (auto gn = pn["grpc"]) {
            cfg.publishers.grpc.enabled        = gn["enabled"].as<bool>(false);
            cfg.publishers.grpc.port           = gn["port"].as<int>(50051);
            cfg.publishers.grpc.max_connections = gn["max_connections"].as<int>(100);
        }
        if (auto rn = pn["redis"]) {
            cfg.publishers.redis.enabled        = rn["enabled"].as<bool>(false);
            cfg.publishers.redis.host           = rn["host"].as<std::string>("localhost");
            cfg.publishers.redis.port           = rn["port"].as<int>(6379);
            cfg.publishers.redis.stream_prefix  = rn["stream_prefix"].as<std::string>("inference");
            cfg.publishers.redis.max_len        = rn["max_len"].as<int>(1000);
            cfg.publishers.redis.queue_capacity = rn["queue_capacity"].as<int>(10000);
        }
    } else if (auto kn = root["kafka"]) {
        parseKafkaNode(kn, cfg.publishers.kafka);
    }
    if (!cfg.publishers.anyEnabled()) {
        throw std::runtime_error("At least one publisher must be enabled under publishers:");
    }
    if (auto an = root["frame_archive"]) {
        cfg.frame_archive.enabled        = an["enabled"].as<bool>(false);
        cfg.frame_archive.allow_gpu_frames = an["allow_gpu_frames"].as<bool>(true);
        cfg.frame_archive.worker_count   = an["worker_count"].as<int>(1);
        cfg.frame_archive.local_dir      = an["local_dir"].as<std::string>("./data/frames");
        cfg.frame_archive.save_interval  = an["save_interval"].as<int>(1);
        cfg.frame_archive.jpeg_quality   = an["jpeg_quality"].as<int>(90);
        cfg.frame_archive.queue_capacity = an["queue_capacity"].as<int>(4096);
        if (cfg.frame_archive.worker_count <= 0) throw std::runtime_error("frame_archive.worker_count must be >= 1");
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
    if (const char* env_repo = std::getenv("INFER_MODEL_REPOSITORY")) {
        if (env_repo[0] != '\0') cfg.server.model_repository = env_repo;
    }
    if (!cfg.server.model_repository.empty()) {
        std::vector<ModelConfig> repo_models = scanModelRepository(cfg.server.model_repository);
        std::unordered_set<std::string> seen;
        for (const auto& m : cfg.models) seen.insert(m.id);
        for (auto& rm : repo_models) {
            if (seen.count(rm.id)) {
                throw std::runtime_error("duplicate model id between root config models and model_repository: " + rm.id);
            }
            seen.insert(rm.id);
            cfg.models.push_back(std::move(rm));
        }
    }
    validateAppConfig(cfg);
    return cfg;
}

} // namespace infer
