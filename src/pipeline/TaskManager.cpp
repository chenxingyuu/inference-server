#include "pipeline/TaskManager.h"
#include "pipeline/StageFactory.h"
#include "stream/GpuDeviceAllocator.h"
#include "stream/StreamHealthRegistry.h"
#include "common/Logger.h"

#include <algorithm>
#include <set>

namespace infer {

namespace {

const char* deviceTypeStr(DeviceType d) {
    switch (d) {
        case DeviceType::CUDA:   return "tensorrt";
        case DeviceType::Ascend: return "ascend";
        case DeviceType::CPU:    return "cpu";
        case DeviceType::MPS:    return "mps";
    }
    return "unknown";
}

const char* yoloVersionStr(YOLOVersion v) {
    switch (v) {
        case YOLOVersion::v5:      return "v5";
        case YOLOVersion::v8:      return "v8";
        case YOLOVersion::v11:     return "v11";
        case YOLOVersion::v26:     return "v26";
        case YOLOVersion::Unknown: return "unknown";
    }
    return "unknown";
}

std::string findInferModelId(const PipelineConfig& pipeline) {
    for (const auto& node : pipeline.nodes) {
        if (node.type != "infer.engine") continue;
        auto it = node.with.find("model_id");
        if (it != node.with.end()) return it->second;
    }
    return {};
}

} // namespace

TaskManager::TaskManager(AppConfig cfg,
                         std::string config_path,
                         IPublisher& publisher,
                         std::shared_ptr<FrameArchiver> frame_archiver)
    : cfg_(std::move(cfg))
    , config_path_(std::move(config_path))
    , publisher_(publisher)
    , frame_archiver_(std::move(frame_archiver))
{
    runtime_state_ = loadRuntimeState(runtimeStatePath(config_path_));
    applyRuntimeState(cfg_, runtime_state_);
}

void TaskManager::loadAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Running) {
            entry.executor->stop();
            entry.state = State::Stopped;
        }
    }
    entries_.clear();

    std::shared_ptr<GpuDeviceAllocator> gpu_alloc;
    {
        std::set<int> device_id_set;
        for (const auto& model : cfg_.models) {
            if (!model.device_ids.empty()) {
                for (int id : model.device_ids) device_id_set.insert(id);
            } else {
                device_id_set.insert(model.device_id);
            }
        }
        if (!device_id_set.empty()) {
            gpu_alloc = std::make_shared<GpuDeviceAllocator>(
                std::vector<int>(device_id_set.begin(), device_id_set.end()));
        }
    }

    for (const auto& task : cfg_.tasks) {
        buildEntry(task, false);
    }
    (void)gpu_alloc; // used inside buildEntry via StageFactory
}

bool TaskManager::buildEntry(const TaskConfig& task, bool autostart) {
    auto source       = cfg_.findSource(task.source_id);
    auto pipeline_tpl = cfg_.findPipeline(task.pipeline_id);
    if (!source || !pipeline_tpl) return false;

    bool use_ascend_dvpp = task.use_ascend_dvpp;
    if (use_ascend_dvpp) {
        const std::string infer_model_id = findInferModelId(*pipeline_tpl);
        const auto* model_cfg = infer_model_id.empty() ? nullptr : cfg_.findModel(infer_model_id);
        if (!model_cfg || model_cfg->backend != DeviceType::Ascend) {
            LOG_WARN("TaskManager: task {} use_ascend_dvpp but no Ascend model; fallback", task.id);
            use_ascend_dvpp = false;
        }
    }

    PipelineConfig runtime_cfg = *pipeline_tpl;
    runtime_cfg.id = task.id;
    auto executor = std::make_shared<GraphExecutor>(runtime_cfg);
    StageFactory::Context ctx{
        cfg_,
        *source,
        publisher_,
        frame_archiver_,
        task.sample_fps,
        task.sampling_mode,
        task.use_hwdec,
        use_ascend_dvpp,
        task.ascend_device_id,
        nullptr};
    for (const auto& node : runtime_cfg.nodes)
        executor->addStage(StageFactory::create(node, ctx));
    executor->build();
    LOG_INFO("Pipeline built for task={} (pipeline={}, source={})\n{}",
             task.id, task.pipeline_id, task.source_id, executor->graphText());

    if (autostart) executor->start();
    entries_[task.id] = Entry{std::move(executor), autostart ? State::Running : State::Stopped};
    return true;
}

void TaskManager::startAll() {
    std::vector<std::shared_ptr<GraphExecutor>> to_start;
    {
        std::lock_guard<std::mutex> lock(mu_);
        to_start.reserve(entries_.size());
        for (auto& [_, entry] : entries_) {
            if (entry.state == State::Running) continue;
            entry.state = State::Running;
            to_start.push_back(entry.executor);
        }
    }
    for (const auto& executor : to_start) {
        executor->start();
    }
}

void TaskManager::stopAll() {
    std::vector<std::shared_ptr<GraphExecutor>> to_stop;
    {
        std::lock_guard<std::mutex> lock(mu_);
        to_stop.reserve(entries_.size());
        for (auto& [_, entry] : entries_) {
            if (entry.state == State::Stopped) continue;
            entry.state = State::Stopped;
            to_stop.push_back(entry.executor);
        }
    }
    for (const auto& executor : to_stop) {
        executor->stop();
    }
}

bool TaskManager::start(const std::string& task_id) {
    std::shared_ptr<GraphExecutor> executor;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(task_id);
        if (it == entries_.end()) return false;
        if (it->second.state == State::Stopped) {
            it->second.state = State::Running;
            executor = it->second.executor;
        }
    }
    if (executor) {
        executor->start();
    }
    return true;
}

bool TaskManager::stop(const std::string& task_id) {
    std::shared_ptr<GraphExecutor> executor;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(task_id);
        if (it == entries_.end()) return false;
        if (it->second.state == State::Running) {
            it->second.state = State::Stopped;
            executor = it->second.executor;
        }
    }
    if (executor) {
        executor->stop();
    }
    return true;
}

std::vector<std::pair<std::string, TaskManager::State>> TaskManager::listTasks() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<std::string, State>> out;
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) out.emplace_back(id, entry.state);
    return out;
}

std::vector<SourceInfo> TaskManager::listSources() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SourceInfo> out;
    out.reserve(cfg_.sources.size());
    for (const auto& src : cfg_.sources) {
        const auto h = StreamHealthRegistry::get().getHealth(src.id);
        out.push_back({src.id, src.url, streamStateStr(h.state), static_cast<int>(h.reconnect_count)});
    }
    return out;
}

std::vector<PipelineInfo> TaskManager::listPipelines() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<PipelineInfo> out;
    out.reserve(cfg_.pipelines.size());
    for (const auto& p : cfg_.pipelines) {
        PipelineInfo info;
        info.id = p.id;
        info.edge_count = static_cast<int>(p.edges.size());
        for (const auto& node : p.nodes) info.nodes.push_back(node.id);
        out.push_back(std::move(info));
    }
    return out;
}

std::vector<ModelInfo> TaskManager::listModels() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ModelInfo> out;
    out.reserve(cfg_.models.size());
    for (const auto& m : cfg_.models) {
        const auto& s = m.input_shape;
        std::string shape = std::to_string(s.channels) + "x"
                          + std::to_string(s.height)   + "x"
                          + std::to_string(s.width);
        out.push_back({m.id, deviceTypeStr(m.backend), yoloVersionStr(m.version),
                       std::move(shape), m.batch_size, m.instance_count});
    }
    return out;
}

// ── Mutation helpers ──────────────────────────────────────────────────────────

void TaskManager::persist() {
    // Rebuild runtime_state_ from current cfg_ delta vs initial.
    // Simpler: treat runtime_state_ as the authoritative record (maintained in each mutator).
    saveRuntimeState(runtime_state_, runtimeStatePath(config_path_));
}

bool TaskManager::addSource(const PipelineSourceConfig& src) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cfg_.findSource(src.id)) return false;
    cfg_.sources.push_back(src);
    // Remove from removed list if present, add to added list.
    auto& removed = runtime_state_.removed_source_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), src.id), removed.end());
    runtime_state_.added_sources.push_back(src);
    persist();
    return true;
}

bool TaskManager::removeSource(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!cfg_.findSource(id)) return false;
    // Check no running task uses this source.
    for (const auto& t : cfg_.tasks)
        if (t.source_id == id) return false;
    cfg_.sources.erase(std::remove_if(cfg_.sources.begin(), cfg_.sources.end(),
        [&](const auto& s){ return s.id == id; }), cfg_.sources.end());
    // Update runtime state.
    auto& added = runtime_state_.added_sources;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& s){ return s.id == id; }), added.end());
    runtime_state_.removed_source_ids.push_back(id);
    persist();
    return true;
}

bool TaskManager::addPipeline(const PipelineConfig& pipeline) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cfg_.findPipeline(pipeline.id)) return false;
    cfg_.pipelines.push_back(pipeline);
    auto& removed = runtime_state_.removed_pipeline_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), pipeline.id), removed.end());
    runtime_state_.added_pipelines.push_back(pipeline);
    persist();
    return true;
}

bool TaskManager::removePipeline(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!cfg_.findPipeline(id)) return false;
    for (const auto& t : cfg_.tasks)
        if (t.pipeline_id == id) return false;
    cfg_.pipelines.erase(std::remove_if(cfg_.pipelines.begin(), cfg_.pipelines.end(),
        [&](const auto& p){ return p.id == id; }), cfg_.pipelines.end());
    auto& added = runtime_state_.added_pipelines;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& p){ return p.id == id; }), added.end());
    runtime_state_.removed_pipeline_ids.push_back(id);
    persist();
    return true;
}

bool TaskManager::addTask(const TaskConfig& task) {
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.count(task.id)) return false;
    if (!buildEntry(task, true)) return false;
    cfg_.tasks.push_back(task);
    auto& removed = runtime_state_.removed_task_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), task.id), removed.end());
    runtime_state_.added_tasks.push_back(task);
    persist();
    return true;
}

bool TaskManager::removeTask(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Running) {
        it->second.executor->stop();
    }
    entries_.erase(it);
    cfg_.tasks.erase(std::remove_if(cfg_.tasks.begin(), cfg_.tasks.end(),
        [&](const auto& t){ return t.id == id; }), cfg_.tasks.end());
    auto& added = runtime_state_.added_tasks;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& t){ return t.id == id; }), added.end());
    runtime_state_.removed_task_ids.push_back(id);
    persist();
    return true;
}

bool TaskManager::loadModel(const ModelConfig& model) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cfg_.findModel(model.id)) return false;
    cfg_.models.push_back(model);
    auto& removed = runtime_state_.removed_model_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), model.id), removed.end());
    runtime_state_.added_models.push_back(model);
    persist();
    return true;
}

bool TaskManager::unloadModel(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!cfg_.findModel(id)) return false;
    // Check no running task uses this model (via pipeline infer.engine node).
    for (const auto& [tid, entry] : entries_) {
        if (entry.state != State::Running) continue;
        const auto* task = cfg_.findTask(tid);
        if (!task) continue;
        const auto* pl = cfg_.findPipeline(task->pipeline_id);
        if (!pl) continue;
        if (findInferModelId(*pl) == id) return false;
    }
    cfg_.models.erase(std::remove_if(cfg_.models.begin(), cfg_.models.end(),
        [&](const auto& m){ return m.id == id; }), cfg_.models.end());
    auto& added = runtime_state_.added_models;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& m){ return m.id == id; }), added.end());
    runtime_state_.removed_model_ids.push_back(id);
    persist();
    return true;
}

} // namespace infer
