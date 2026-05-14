#include "common/RuntimeState.h"
#include "common/Config.h"
#include "common/Logger.h"

#include <nlohmann/json.hpp>

#include <fstream>

namespace infer {

using json = nlohmann::json;

std::string runtimeStatePath(const std::string& config_path) {
    return config_path + ".runtime.json";
}

// ── Serialization helpers ─────────────────────────────────────────────────────

static json toJson(const PipelineSourceConfig& s) {
    return {
        {"id",                    s.id},
        {"url",                   s.url},
        {"reconnect_delay_ms",    s.reconnect_delay_ms},
        {"max_reconnect_delay_ms",s.max_reconnect_delay_ms},
        {"degraded_threshold",    s.degraded_threshold},
        {"max_reconnect_attempts",s.max_reconnect_attempts},
        {"open_timeout_ms",       s.open_timeout_ms},
        {"read_timeout_ms",       s.read_timeout_ms},
    };
}

static PipelineSourceConfig sourceFromJson(const json& j) {
    PipelineSourceConfig s;
    s.id                     = j.value("id",                    s.id);
    s.url                    = j.value("url",                   s.url);
    s.reconnect_delay_ms     = j.value("reconnect_delay_ms",    s.reconnect_delay_ms);
    s.max_reconnect_delay_ms = j.value("max_reconnect_delay_ms",s.max_reconnect_delay_ms);
    s.degraded_threshold     = j.value("degraded_threshold",    s.degraded_threshold);
    s.max_reconnect_attempts = j.value("max_reconnect_attempts",s.max_reconnect_attempts);
    s.open_timeout_ms        = j.value("open_timeout_ms",       s.open_timeout_ms);
    s.read_timeout_ms        = j.value("read_timeout_ms",       s.read_timeout_ms);
    return s;
}

static json toJson(const StageConfig& n) {
    json j = {{"id", n.id}, {"type", n.type}};
    if (!n.with.empty()) j["with"] = n.with;
    return j;
}
static StageConfig stageFromJson(const json& j) {
    StageConfig n;
    n.id   = j.value("id",   std::string{});
    n.type = j.value("type", std::string{});
    if (j.contains("with") && j["with"].is_object())
        for (const auto& [k, v] : j["with"].items())
            n.with[k] = v.get<std::string>();
    return n;
}

static std::string yoloVersionToStr(YOLOVersion v) {
    switch (v) {
        case YOLOVersion::v5:  return "yolov5";
        case YOLOVersion::v8:  return "yolov8";
        case YOLOVersion::v11: return "yolo11";
        case YOLOVersion::v26: return "yolo26";
        default:               return "yolov8";
    }
}

static std::string samplingModeToStr(SamplingMode m) {
    return m == SamplingMode::TimeBased ? "time_based" : "frame_count";
}

static json toJson(const CascadeConfig& c) {
    json tc = json::array();
    for (int cls : c.trigger_classes) tc.push_back(cls);
    return {
        {"model_id",        c.model_id},
        {"trigger_classes", tc},
        {"crop_expand",     c.crop_expand},
        {"attribute_key",   c.attribute_key},
    };
}
static CascadeConfig cascadeFromJson(const json& j) {
    CascadeConfig c;
    c.model_id      = j.value("model_id",      c.model_id);
    c.crop_expand   = j.value("crop_expand",   c.crop_expand);
    c.attribute_key = j.value("attribute_key", c.attribute_key);
    for (const auto& v : j.value("trigger_classes", json::array()))
        c.trigger_classes.push_back(v.get<int>());
    return c;
}

static std::string edgeDropPolicyToJson(EdgeDropPolicy p) {
    switch (p) {
        case EdgeDropPolicy::DropOldest: return "drop_oldest";
        case EdgeDropPolicy::DropNewest: return "drop_newest";
        default:                         return "block";
    }
}

static json toJson(const EdgeConfig& e) {
    return {{"from", e.from}, {"to", e.to},
            {"capacity", e.capacity},
            {"drop_policy", edgeDropPolicyToJson(e.drop_policy)}};
}
static EdgeConfig edgeFromJson(const json& j) {
    EdgeConfig e;
    e.from     = j.value("from",     std::string{});
    e.to       = j.value("to",       std::string{});
    e.capacity = j.value("capacity", e.capacity);
    if (const auto it = j.find("drop_policy"); it != j.end() && it->is_string()) {
        const std::string s = it->get<std::string>();
        if (!s.empty()) {
            try {
                e.drop_policy = parseEdgeDropPolicy(s);
            } catch (const std::exception& ex) {
                LOG_WARN("RuntimeState: invalid drop_policy '{}': {}", s, ex.what());
            }
        }
    }
    return e;
}

static json toJson(const PipelineConfig& p) {
    json nodes = json::array();
    for (const auto& n : p.nodes) nodes.push_back(toJson(n));
    json edges = json::array();
    for (const auto& e : p.edges) edges.push_back(toJson(e));
    return {{"id", p.id}, {"nodes", nodes}, {"edges", edges}};
}
static PipelineConfig pipelineFromJson(const json& j) {
    PipelineConfig p;
    p.id = j.value("id", std::string{});
    for (const auto& n : j.value("nodes", json::array())) p.nodes.push_back(stageFromJson(n));
    for (const auto& e : j.value("edges", json::array())) p.edges.push_back(edgeFromJson(e));
    return p;
}

static json toJson(const TaskConfig& t) {
    return {
        {"id",               t.id},
        {"source_id",        t.source_id},
        {"pipeline_id",      t.pipeline_id},
        {"sample_fps",       t.sample_fps},
        {"sampling_mode",    samplingModeToStr(t.sampling_mode)},
        {"use_hwdec",        t.use_hwdec},
        {"use_ascend_dvpp",  t.use_ascend_dvpp},
        {"ascend_device_id", t.ascend_device_id},
    };
}
static TaskConfig taskFromJson(const json& j) {
    TaskConfig t;
    t.id               = j.value("id",               t.id);
    t.source_id        = j.value("source_id",        t.source_id);
    t.pipeline_id      = j.value("pipeline_id",      t.pipeline_id);
    t.sample_fps       = j.value("sample_fps",       t.sample_fps);
    t.sampling_mode    = parseSamplingMode(j.value("sampling_mode", std::string{"frame_count"}));
    t.use_hwdec        = j.value("use_hwdec",        t.use_hwdec);
    t.use_ascend_dvpp  = j.value("use_ascend_dvpp",  t.use_ascend_dvpp);
    t.ascend_device_id = j.value("ascend_device_id", t.ascend_device_id);
    return t;
}

static json toJson(const ModelConfig& m) {
    json cn = json::array();
    for (const auto& s : m.class_names) cn.push_back(s);
    json di = json::array();
    for (int id : m.device_ids) di.push_back(id);
    json pbs = json::array();
    for (int bs : m.preferred_batch_sizes) pbs.push_back(bs);
    json om = json::object();
    for (const auto& [k, v] : m.om_paths) om[std::to_string(k)] = v;
    json casc = json::array();
    for (const auto& c : m.cascade) casc.push_back(toJson(c));
    return {
        {"id",                   m.id},
        {"version",              yoloVersionToStr(m.version)},
        {"backend",              deviceTypeToStr(m.backend)},
        {"model_type",           m.model_type == ModelType::Classifier ? "classifier" : "detector"},
        {"onnx_path",            m.onnx_path},
        {"engine_path",          m.engine_path},
        {"om_paths",             om},
        {"batch_size",           m.batch_size},
        {"input_shape",          {{"c", m.input_shape.channels},
                                  {"h", m.input_shape.height},
                                  {"w", m.input_shape.width}}},
        {"conf_thresh",          m.conf_thresh},
        {"nms_thresh",           m.nms_thresh},
        {"device_id",            m.device_id},
        {"num_classes",          m.num_classes},
        {"class_names",          cn},
        {"buffer_pool_size",     m.buffer_pool_size},
        {"instance_count",       m.instance_count},
        {"device_ids",           di},
        {"cascade",              casc},
        {"preferred_batch_sizes",pbs},
        {"max_queue_delay_us",   m.max_queue_delay_us},
    };
}
static ModelConfig modelFromJson(const json& j) {
    ModelConfig m;
    m.id             = j.value("id",           m.id);
    m.version        = parseYOLOVersion(j.value("version", std::string{"yolov8"}));
    m.backend        = parseDeviceType(j.value("backend",  std::string{"cpu"}));
    m.model_type     = j.value("model_type", std::string{"detector"}) == "classifier"
                           ? ModelType::Classifier : ModelType::Detector;
    m.onnx_path      = j.value("onnx_path",    m.onnx_path);
    m.engine_path    = j.value("engine_path",  m.engine_path);
    m.batch_size     = j.value("batch_size",   m.batch_size);
    m.conf_thresh    = j.value("conf_thresh",  m.conf_thresh);
    m.nms_thresh     = j.value("nms_thresh",   m.nms_thresh);
    m.device_id      = j.value("device_id",    m.device_id);
    m.num_classes    = j.value("num_classes",  m.num_classes);
    m.buffer_pool_size    = j.value("buffer_pool_size",   m.buffer_pool_size);
    m.instance_count      = j.value("instance_count",     m.instance_count);
    m.max_queue_delay_us  = j.value("max_queue_delay_us", m.max_queue_delay_us);
    if (j.contains("input_shape") && j["input_shape"].is_object()) {
        m.input_shape.channels = j["input_shape"].value("c", m.input_shape.channels);
        m.input_shape.height   = j["input_shape"].value("h", m.input_shape.height);
        m.input_shape.width    = j["input_shape"].value("w", m.input_shape.width);
    }
    if (j.contains("om_paths") && j["om_paths"].is_object())
        for (const auto& [k, v] : j["om_paths"].items())
            m.om_paths[std::stoi(k)] = v.get<std::string>();
    for (const auto& v : j.value("class_names",          json::array())) m.class_names.push_back(v.get<std::string>());
    for (const auto& v : j.value("device_ids",           json::array())) m.device_ids.push_back(v.get<int>());
    for (const auto& v : j.value("preferred_batch_sizes", json::array())) m.preferred_batch_sizes.push_back(v.get<int>());
    for (const auto& v : j.value("cascade",              json::array())) m.cascade.push_back(cascadeFromJson(v));
    return m;
}

// ── Public API ────────────────────────────────────────────────────────────────

RuntimeState loadRuntimeState(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        LOG_WARN("RuntimeState: ignoring malformed {}: {}", path, e.what());
        return {};
    }

    RuntimeState s;
    for (const auto& v : j.value("added_sources",    json::array())) s.added_sources.push_back(sourceFromJson(v));
    for (const auto& v : j.value("removed_source_ids",json::array())) s.removed_source_ids.push_back(v.get<std::string>());
    for (const auto& v : j.value("added_pipelines",  json::array())) s.added_pipelines.push_back(pipelineFromJson(v));
    for (const auto& v : j.value("removed_pipeline_ids",json::array())) s.removed_pipeline_ids.push_back(v.get<std::string>());
    for (const auto& v : j.value("added_tasks",      json::array())) s.added_tasks.push_back(taskFromJson(v));
    for (const auto& v : j.value("removed_task_ids", json::array())) s.removed_task_ids.push_back(v.get<std::string>());
    for (const auto& v : j.value("added_models",     json::array())) s.added_models.push_back(modelFromJson(v));
    for (const auto& v : j.value("removed_model_ids",json::array())) s.removed_model_ids.push_back(v.get<std::string>());
    return s;
}

void saveRuntimeState(const RuntimeState& state, const std::string& path) {
    json j;

    auto& as = j["added_sources"] = json::array();
    for (const auto& v : state.added_sources) as.push_back(toJson(v));
    auto& rs = j["removed_source_ids"] = json::array();
    for (const auto& v : state.removed_source_ids) rs.push_back(v);

    auto& ap = j["added_pipelines"] = json::array();
    for (const auto& v : state.added_pipelines) ap.push_back(toJson(v));
    auto& rp = j["removed_pipeline_ids"] = json::array();
    for (const auto& v : state.removed_pipeline_ids) rp.push_back(v);

    auto& at = j["added_tasks"] = json::array();
    for (const auto& v : state.added_tasks) at.push_back(toJson(v));
    auto& rt = j["removed_task_ids"] = json::array();
    for (const auto& v : state.removed_task_ids) rt.push_back(v);

    auto& am = j["added_models"] = json::array();
    for (const auto& v : state.added_models) am.push_back(toJson(v));
    auto& rm = j["removed_model_ids"] = json::array();
    for (const auto& v : state.removed_model_ids) rm.push_back(v);

    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f.is_open()) {
            LOG_ERROR("RuntimeState: cannot write to {}", tmp);
            return;
        }
        f << j.dump(2);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        LOG_ERROR("RuntimeState: rename {} → {} failed", tmp, path);
}

void applyRuntimeState(AppConfig& cfg, const RuntimeState& state) {
    // Helper: erase by id from a vector of structs with .id member
    auto eraseById = [](auto& vec, const std::vector<std::string>& ids) {
        for (const auto& id : ids)
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [&](const auto& e) { return e.id == id; }), vec.end());
    };
    auto hasId = [](const auto& vec, const std::string& id) {
        for (const auto& e : vec) if (e.id == id) return true;
        return false;
    };

    eraseById(cfg.sources,   state.removed_source_ids);
    eraseById(cfg.pipelines, state.removed_pipeline_ids);
    eraseById(cfg.tasks,     state.removed_task_ids);
    eraseById(cfg.models,    state.removed_model_ids);

    for (const auto& s : state.added_sources)
        if (!hasId(cfg.sources, s.id)) cfg.sources.push_back(s);
    for (const auto& p : state.added_pipelines)
        if (!hasId(cfg.pipelines, p.id)) cfg.pipelines.push_back(p);
    for (const auto& t : state.added_tasks)
        if (!hasId(cfg.tasks, t.id)) cfg.tasks.push_back(t);
    for (const auto& m : state.added_models)
        if (!hasId(cfg.models, m.id)) cfg.models.push_back(m);
}

} // namespace infer
