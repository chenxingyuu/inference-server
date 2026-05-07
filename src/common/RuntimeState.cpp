#include "common/RuntimeState.h"
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

static json toJson(const EdgeConfig& e) {
    return {{"from", e.from}, {"to", e.to},
            {"capacity", e.capacity}};
}
static EdgeConfig edgeFromJson(const json& j) {
    EdgeConfig e;
    e.from     = j.value("from",     std::string{});
    e.to       = j.value("to",       std::string{});
    e.capacity = j.value("capacity", e.capacity);
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
        {"id",           t.id},
        {"source_id",    t.source_id},
        {"pipeline_id",  t.pipeline_id},
        {"sample_fps",   t.sample_fps},
        {"use_hwdec",    t.use_hwdec},
    };
}
static TaskConfig taskFromJson(const json& j) {
    TaskConfig t;
    t.id          = j.value("id",          t.id);
    t.source_id   = j.value("source_id",   t.source_id);
    t.pipeline_id = j.value("pipeline_id", t.pipeline_id);
    t.sample_fps  = j.value("sample_fps",  t.sample_fps);
    t.use_hwdec   = j.value("use_hwdec",   t.use_hwdec);
    return t;
}

static json toJson(const ModelConfig& m) {
    return {
        {"id",          m.id},
        {"backend",     deviceTypeToStr(m.backend)},
        {"onnx_path",   m.onnx_path},
        {"engine_path", m.engine_path},
        {"batch_size",  m.batch_size},
        {"input_shape", {{"c", m.input_shape.channels},
                         {"h", m.input_shape.height},
                         {"w", m.input_shape.width}}},
    };
}
static ModelConfig modelFromJson(const json& j) {
    ModelConfig m;
    m.id          = j.value("id",          m.id);
    m.backend     = parseDeviceType(j.value("backend", std::string{"cpu"}));
    m.onnx_path   = j.value("onnx_path",   m.onnx_path);
    m.engine_path = j.value("engine_path", m.engine_path);
    m.batch_size  = j.value("batch_size",  m.batch_size);
    if (j.contains("input_shape") && j["input_shape"].is_object()) {
        m.input_shape.channels = j["input_shape"].value("c", m.input_shape.channels);
        m.input_shape.height   = j["input_shape"].value("h", m.input_shape.height);
        m.input_shape.width    = j["input_shape"].value("w", m.input_shape.width);
    }
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
