#include "pipeline/TaskManager.h"
#include "pipeline/StageFactory.h"
#include "stream/GpuDeviceAllocator.h"
#include "stream/StreamHealthRegistry.h"
#include "common/Logger.h"
#include "common/Config.h"

#include <algorithm>
#include <set>
#include <string_view>

namespace infer {

namespace {

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

std::string interpolateOutputUrlPlaceholders(
    std::string_view raw,
    const TaskConfig& task,
    const std::string& pipeline_id,
    const std::string& stage_id) {
    std::string out;
    out.reserve(raw.size() + 16);

    std::size_t pos = 0;
    while (pos < raw.size()) {
        const std::size_t open = raw.find('{', pos);
        if (open == std::string_view::npos) {
            out.append(raw.substr(pos));
            break;
        }
        out.append(raw.substr(pos, open - pos));
        const std::size_t close = raw.find('}', open + 1);
        if (close == std::string_view::npos) {
            throw std::runtime_error(
                "TaskManager: task '" + task.id + "' pipeline '" + pipeline_id + "' stage '" + stage_id +
                "' has unterminated placeholder in sink.stream output_url: " + std::string(raw));
        }
        const std::string token(raw.substr(open + 1, close - open - 1));
        if (token == "task_id") {
            out += task.id;
        } else if (token == "source_id") {
            out += task.source_id;
        } else {
            throw std::runtime_error(
                "TaskManager: task '" + task.id + "' pipeline '" + pipeline_id + "' stage '" + stage_id +
                "' has unsupported placeholder '{" + token +
                "}' in sink.stream output_url (supported: {task_id}, {source_id})");
        }
        pos = close + 1;
    }
    return out;
}

void interpolatePipelineForTask(PipelineConfig& runtime_cfg, const TaskConfig& task, const std::string& pipeline_id) {
    for (auto& node : runtime_cfg.nodes) {
        if (node.type != "sink.stream") continue;
        auto output_url_it = node.with.find("output_url");
        if (output_url_it == node.with.end()) continue;
        const std::string_view raw(output_url_it->second);
        if (raw.find('{') == std::string_view::npos && raw.find('}') == std::string_view::npos) {
            continue;
        }
        output_url_it->second = interpolateOutputUrlPlaceholders(raw, task, pipeline_id, node.id);
    }
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
    int  ingest_vpc_out_w = 0;
    int  ingest_vpc_out_h = 0;
    if (use_ascend_dvpp) {
        const std::string infer_model_id = findInferModelId(*pipeline_tpl);
        const auto* model_cfg = infer_model_id.empty() ? nullptr : cfg_.findModel(infer_model_id);
        if (!model_cfg || model_cfg->backend != DeviceType::Ascend) {
            LOG_WARN("TaskManager: task {} use_ascend_dvpp but no Ascend model; fallback", task.id);
            use_ascend_dvpp = false;
        } else {
            ingest_vpc_out_w = model_cfg->input_shape.width;
            ingest_vpc_out_h = model_cfg->input_shape.height;
        }
    }

    PipelineConfig runtime_cfg = *pipeline_tpl;
    runtime_cfg.id = task.id;
    interpolatePipelineForTask(runtime_cfg, task, pipeline_tpl->id);
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
        ingest_vpc_out_w,
        ingest_vpc_out_h,
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

std::vector<TaskManager::TaskRuntimeInfo> TaskManager::listTasks() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<TaskRuntimeInfo> out;
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        TaskConfig cfg;
        for (const auto& t : cfg_.tasks)
            if (t.id == id) { cfg = t; break; }
        out.push_back({cfg, entry.state});
    }
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
        for (const auto& node : p.nodes) {
            PipelineNodeInfo ni;
            ni.id   = node.id;
            ni.type = node.type;
            ni.with = node.with;
            info.nodes.push_back(std::move(ni));
        }
        for (const auto& e : p.edges) {
            PipelineEdgeInfo ei;
            ei.from        = e.from;
            ei.to          = e.to;
            ei.capacity    = e.capacity;
            switch (e.drop_policy) {
                case EdgeDropPolicy::DropOldest: ei.drop_policy = "drop_oldest"; break;
                case EdgeDropPolicy::DropNewest: ei.drop_policy = "drop_newest"; break;
                default:                         ei.drop_policy = "block";       break;
            }
            info.edges.push_back(std::move(ei));
        }
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
        ModelInfo info;
        info.id                     = m.id;
        info.backend                = deviceTypeToStr(m.backend);
        info.version                = yoloVersionStr(m.version);
        info.input_shape            = std::move(shape);
        info.batch_size             = m.batch_size;
        info.instance_count         = m.instance_count;
        info.model_type             = (m.model_type == ModelType::Classifier) ? "classifier" : "detector";
        info.num_classes            = m.num_classes;
        info.conf_thresh            = m.conf_thresh;
        info.nms_thresh             = m.nms_thresh;
        info.device_id              = m.device_id;
        info.preferred_batch_sizes  = m.preferred_batch_sizes;
        info.max_queue_delay_us     = m.max_queue_delay_us;
        info.class_names            = m.class_names;
        out.push_back(std::move(info));
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

bool TaskManager::updatePipeline(const PipelineConfig& pipeline) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto* old_ptr = cfg_.findPipeline(pipeline.id);
    if (!old_ptr) return false;

    // Block only while a task is actually running this pipeline; stopped tasks keep a
    // snapshot GraphExecutor and must be rebuilt after the template changes.
    for (const auto& t : cfg_.tasks) {
        if (t.pipeline_id != pipeline.id) continue;
        auto it = entries_.find(t.id);
        if (it != entries_.end() && it->second.state == State::Running) return false;
    }

    const PipelineConfig backup = *old_ptr;

    for (auto& p : cfg_.pipelines)
        if (p.id == pipeline.id) { p = pipeline; break; }
    for (auto& p : runtime_state_.added_pipelines)
        if (p.id == pipeline.id) { p = pipeline; break; }

    std::vector<TaskConfig> affected;
    for (const auto& t : cfg_.tasks) {
        if (t.pipeline_id == pipeline.id) affected.push_back(t);
    }
    for (const auto& t : affected) {
        entries_.erase(t.id);
    }

    for (const auto& t : affected) {
        if (!buildEntry(t, false)) {
            LOG_ERROR(
                "TaskManager: updatePipeline failed to rebuild stopped task '{}' after template change; restoring previous pipeline",
                t.id);
            for (auto& p : cfg_.pipelines)
                if (p.id == pipeline.id) { p = backup; break; }
            for (auto& p : runtime_state_.added_pipelines)
                if (p.id == pipeline.id) { p = backup; break; }
            for (const auto& t2 : affected) {
                entries_.erase(t2.id);
            }
            for (const auto& t2 : affected) {
                if (!buildEntry(t2, false)) {
                    LOG_ERROR("TaskManager: updatePipeline failed to restore task '{}' to previous graph", t2.id);
                }
            }
            return false;
        }
    }

    persist();
    return true;
}

bool TaskManager::updateTask(const TaskConfig& task) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(task.id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Running) return false;
    if (!cfg_.findSource(task.source_id))     return false;
    if (!cfg_.findPipeline(task.pipeline_id)) return false;
    entries_.erase(it);
    for (auto& t : cfg_.tasks)
        if (t.id == task.id) { t = task; break; }
    for (auto& t : runtime_state_.added_tasks)
        if (t.id == task.id) { t = task; break; }
    buildEntry(task, false);
    persist();
    return true;
}

bool TaskManager::addTask(const TaskConfig& task, bool autostart) {
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.count(task.id)) return false;
    if (!buildEntry(task, autostart)) return false;
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

std::vector<RepositoryModelInfo> TaskManager::listRepositoryModels() const {
    if (cfg_.server.model_repository.empty()) return {};
    std::vector<ModelConfig> repo;
    try {
        repo = scanModelRepository(cfg_.server.model_repository);
    } catch (const std::exception&) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<RepositoryModelInfo> out;
    out.reserve(repo.size());
    for (const auto& m : repo) {
        RepositoryModelInfo info;
        info.id             = m.id;
        info.backend        = deviceTypeToStr(m.backend);
        info.version        = yoloVersionStr(m.version);
        info.model_type     = (m.model_type == ModelType::Classifier) ? "classifier" : "detector";
        info.batch_size     = m.batch_size;
        info.num_classes    = m.num_classes;
        info.conf_thresh    = m.conf_thresh;
        info.nms_thresh     = m.nms_thresh;
        info.instance_count = m.instance_count;
        info.loaded         = (cfg_.findModel(m.id) != nullptr);
        out.push_back(std::move(info));
    }
    return out;
}

bool TaskManager::loadFromRepository(const std::string& id) {
    if (cfg_.server.model_repository.empty()) return false;
    std::vector<ModelConfig> repo;
    try {
        repo = scanModelRepository(cfg_.server.model_repository);
    } catch (const std::exception&) {
        return false;
    }
    const auto it = std::find_if(repo.begin(), repo.end(),
                                 [&](const auto& m){ return m.id == id; });
    if (it == repo.end()) return false;
    return loadModel(*it);
}

} // namespace infer
