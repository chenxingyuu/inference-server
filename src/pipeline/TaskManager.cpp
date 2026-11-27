#include "pipeline/TaskManager.h"
#include "pipeline/StageFactory.h"
#include <fstream>
#include <regex>
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

void rememberStoppedTask(RuntimeState& state, const std::string& id) {
    auto& ids = state.stopped_task_ids;
    if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
}

bool forgetStoppedTask(RuntimeState& state, const std::string& id) {
    auto& ids = state.stopped_task_ids;
    const auto before = ids.size();
    ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
    return ids.size() != before;
}

bool taskWasManuallyStopped(const RuntimeState& state, const std::string& id) {
    const auto& ids = state.stopped_task_ids;
    return std::find(ids.begin(), ids.end(), id) != ids.end();
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
                         std::unordered_map<std::string, IPublisher*> publishers,
                         std::shared_ptr<FrameArchiver> frame_archiver, ModelProbe model_probe, ModelTest model_test,
                         GraphBuilder graph_builder, std::chrono::milliseconds drain_timeout)
    : cfg_(std::move(cfg))
    , config_path_(std::move(config_path))
    , publishers_(std::move(publishers))
    , frame_archiver_(std::move(frame_archiver))
    , model_probe_(std::move(model_probe))
    , model_test_(std::move(model_test))
    , graph_builder_(std::move(graph_builder))
    , drain_timeout_(drain_timeout)
{
    runtime_state_ = loadRuntimeState(runtimeStatePath(config_path_));
    applyRuntimeState(cfg_, runtime_state_);
    for (const auto& source : cfg_.sources) trackSourceProbeLocked(source);
    watcher_ = std::thread(&TaskManager::watchTasks, this);
}

TaskManager::~TaskManager() {
    stopSourceMonitoring();
    stopAll();
    {
        std::lock_guard lock(mu_);
        watcher_exit_ = true;
    }
    lifecycle_cv_.notify_all();
    watcher_.join();
}

bool TaskManager::validateTask(const TaskConfig& task) const {
    const auto* pipeline = cfg_.findPipeline(task.pipeline_id);
    if (!cfg_.findSource(task.source_id) || !pipeline || referencesUnregisteredModel(*pipeline)) return false;
    auto runtime = *pipeline;
    interpolatePipelineForTask(runtime, task, pipeline->id);
    return true;
}

void TaskManager::loadAll() {
    stopAll();
    std::lock_guard lock(mu_);
    entries_.clear();
    for (const auto& task : cfg_.tasks) {
        if (validateTask(task)) entries_.try_emplace(task.id);
    }
}

std::unique_ptr<GraphExecutor> TaskManager::buildGraph(const AppConfig& cfg, const TaskConfig& task) {
    const auto* source = cfg.findSource(task.source_id);
    const auto* pipeline_tpl = cfg.findPipeline(task.pipeline_id);
    if (!source || !pipeline_tpl) throw std::runtime_error("task dependency missing");

    bool use_ascend_dvpp = task.use_ascend_dvpp;
    int  ingest_vpc_out_w = 0;
    int  ingest_vpc_out_h = 0;
    if (use_ascend_dvpp) {
        const std::string infer_model_id = findInferModelId(*pipeline_tpl);
        const auto* model_cfg = infer_model_id.empty() ? nullptr : cfg.findModel(infer_model_id);
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
    auto executor = std::make_unique<GraphExecutor>(runtime_cfg);
    StageFactory::Context ctx{
        cfg,
        *source,
        publishers_,
        frame_archiver_,
        task.sample_fps,
        task.sampling_mode,
        task.use_hwdec,
        use_ascend_dvpp,
        task.ascend_device_id,
        ingest_vpc_out_w,
        ingest_vpc_out_h,
        nullptr};
    for (const auto& node : runtime_cfg.nodes) {
        executor->addStage(StageFactory::create(node, ctx));
    }
    executor->build();
    LOG_INFO("Pipeline built for task={} (pipeline={}, source={})\n{}",
             task.id, task.pipeline_id, task.source_id, executor->graphText());

    return executor;
}

void TaskManager::startAll() {
    std::lock_guard lock(mu_);
    for (const auto& [id, entry] : entries_) {
        // Desired running state survives restart. Explicit stop is recorded in
        // runtime state; failure and natural EOF leave the previous desire intact.
        if (!entry.removed && !taskWasManuallyStopped(runtime_state_, id)) startLocked(id);
    }
}

void TaskManager::stopAll() {
    std::unique_lock lock(mu_);
    stopping_all_ = true;
    for (auto& [_, entry] : entries_) {
        if (entry.run) entry.run->cancelled.store(true);
    }
    lifecycle_cv_.notify_all();
    lifecycle_cv_.wait(lock, [&] {
        return std::none_of(entries_.begin(), entries_.end(), [](const auto& item) {
            return item.second.run != nullptr;
        });
    });
    stopping_all_ = false;
}

ITaskManager::ActionResult TaskManager::startLocked(const std::string& task_id) {
    const auto it = entries_.find(task_id);
    if (it == entries_.end() || it->second.removed) return ActionResult::NotFound;
    auto& entry = it->second;
    if (stopping_all_ || (entry.run && entry.run->cancelled.load())) return ActionResult::Busy;
    if (entry.run) return ActionResult::Accepted;

    const auto* config = cfg_.findTask(task_id);
    if (!config) return ActionResult::NotFound;
    auto run = std::make_shared<Run>();
    run->startup_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(cfg_.server.task_start_timeout_ms);
    entry.run = run;
    entry.state = State::Starting;
    entry.failure.reset();
    try {
        run->thread = std::thread([this, task = *config, cfg = cfg_, run] {
            runTask(task, cfg, run);
        });
    } catch (const std::exception& error) {
        LOG_ERROR("Task {} could not start a lifecycle thread: {}", task_id, error.what());
        entry.state = State::Failed;
        entry.failure = TaskFailure{"startup_failed", "Task worker could not be started"};
        entry.run.reset();
    }
    lifecycle_cv_.notify_all();
    return ActionResult::Accepted;
}

ITaskManager::ActionResult TaskManager::requestStart(const std::string& task_id) {
    std::lock_guard lock(mu_);
    const auto result = startLocked(task_id);
    if (result == ActionResult::Accepted && forgetStoppedTask(runtime_state_, task_id)) persist();
    return result;
}

bool TaskManager::start(const std::string& task_id) {
    return requestStart(task_id) == ActionResult::Accepted;
}

bool TaskManager::stop(const std::string& task_id) {
    std::lock_guard lock(mu_);
    const auto it = entries_.find(task_id);
    if (it == entries_.end() || it->second.removed) return false;
    rememberStoppedTask(runtime_state_, task_id);
    persist();
    auto& entry = it->second;
    if (entry.run) {
        entry.run->stop_requested = true;
        entry.run->cancelled.store(true);
        lifecycle_cv_.notify_all();
    } else {
        entry.state = State::Stopped;
        entry.failure.reset();
    }
    return true;
}

void TaskManager::runTask(const TaskConfig& task, const AppConfig& cfg, const std::shared_ptr<Run>& run) {
    std::unique_ptr<GraphExecutor> executor;
    std::optional<TaskFailure> failure;
    try {
        if (!run->cancelled.load()) {
            executor = graph_builder_ ? graph_builder_(cfg, task) : buildGraph(cfg, task);
            executor->start(&run->cancelled);
            std::optional<std::chrono::steady_clock::time_point> drain_deadline;
            while (!run->cancelled.load()) {
                if ((failure = executor->failure())) break;
                std::unique_lock lock(mu_);
                auto& entry = entries_.at(task.id);
                if (run->cancelled.load()) break;
                const auto now = std::chrono::steady_clock::now();
                if (entry.state == State::Starting) {
                    if (now >= run->startup_deadline) {
                        recordStartupTimeoutLocked(task.id, entry);
                        break;
                    }
                    if (executor->inputsReady()) entry.state = State::Running;
                }
                if (executor->inputsEnded() && !drain_deadline) {
                    drain_deadline = now + drain_timeout_;
                }
                if (executor->drained()) break;
                if (drain_deadline && now >= *drain_deadline) {
                    LOG_ERROR("Task {} output drain timed out", task.id);
                    failure = TaskFailure{"drain_timeout", "Pipeline did not finish processing the video in time"};
                    break;
                }
                lifecycle_cv_.wait_for(lock, std::chrono::milliseconds(10));
            }
        }
    } catch (const std::exception& error) {
        LOG_ERROR("Task {} startup exception: {}", task.id, error.what());
        failure = TaskFailure{"startup_failed", "Pipeline failed to initialize"};
    } catch (...) {
        LOG_ERROR("Task {} encountered an unknown exception during startup", task.id);
        failure = TaskFailure{"startup_failed", "Pipeline failed to initialize"};
    }
    {
        std::lock_guard lock(mu_);
        auto& entry = entries_.at(task.id);
        if (!run->cancelled.exchange(true) && failure) {
            entry.state = State::Failed;
            entry.failure = std::move(failure);
        }
    }
    executor.reset();
    {
        std::lock_guard lock(mu_);
        run->done = true;
    }
    lifecycle_cv_.notify_all();
}

void TaskManager::recordStartupTimeoutLocked(const std::string& task_id, Entry& entry) {
    entry.run->cancelled.store(true);
    LOG_ERROR("Task {} startup timed out", task_id);
    entry.state = State::Failed;
    entry.failure = TaskFailure{"startup_timeout", "Pipeline startup timed out"};
    lifecycle_cv_.notify_all();
}

void TaskManager::watchTasks() {
    std::unique_lock lock(mu_);
    while (!watcher_exit_) {
        const auto now = std::chrono::steady_clock::now();
        std::vector<std::pair<std::string, std::shared_ptr<Run>>> finished;
        for (auto& [id, entry] : entries_) {
            if (!entry.run) continue;
            auto& run = *entry.run;
            if (run.done) {
                finished.emplace_back(id, entry.run);
            } else if (entry.state == State::Starting && now >= run.startup_deadline &&
                       !run.cancelled.load()) {
                recordStartupTimeoutLocked(id, entry);
            }
        }
        if (!finished.empty()) {
            lock.unlock();
            for (const auto& [_, run] : finished) run->thread.join();
            lock.lock();
            for (const auto& [id, run] : finished) {
                const auto it = entries_.find(id);
                if (it == entries_.end() || it->second.run != run) continue;
                auto& entry = it->second;
                if (entry.removed) {
                    entries_.erase(it);
                } else {
                    entry.run.reset();
                    if (entry.state != State::Failed || run->stop_requested) {
                        entry.state = State::Stopped;
                        entry.failure.reset();
                    }
                }
            }
            lifecycle_cv_.notify_all();
        }
        lifecycle_cv_.wait_for(lock, std::chrono::milliseconds(10));
    }
}

std::vector<TaskManager::TaskRuntimeInfo> TaskManager::listTasks() const {
    std::lock_guard lock(mu_);
    std::vector<TaskRuntimeInfo> out;
    out.reserve(entries_.size());
    for (const auto& task : cfg_.tasks) {
        const auto it = entries_.find(task.id);
        if (it == entries_.end() || it->second.removed) continue;
        const auto& entry = it->second;
        out.push_back({task, entry.state, entry.failure, entry.run && entry.run->cancelled.load()});
    }
    return out;
}

std::vector<SourceInfo> TaskManager::listSources() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SourceInfo> out;
    out.reserve(cfg_.sources.size());
    for (const auto& src : cfg_.sources) {
        out.push_back(sourceInfoLocked(src));
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

namespace {

ModelInfo toModelInfo(const ModelConfig& m) {
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
    return info;
}

const std::string* inferModelId(const StageConfig& node) {
    if (node.type != "infer.engine") return nullptr;
    const auto it = node.with.find("model_id");
    return it == node.with.end() ? nullptr : &it->second;
}

} // namespace

std::vector<ModelInfo> TaskManager::listModels() const {
    std::lock_guard lock(mu_);
    std::vector<ModelInfo> out;
    out.reserve(cfg_.models.size());
    for (const auto& model : cfg_.models) out.push_back(toModelInfo(model));
    return out;
}

ModelReferences TaskManager::modelReferences(const std::string& id) const {
    ModelReferences references;
    for (const auto& model : cfg_.models) {
        if (model.id != id && std::any_of(model.cascade.begin(), model.cascade.end(),
                [&](const auto& child) { return child.model_id == id; })) {
            references.cascade_parents.push_back(model.id);
        }
    }
    for (const auto& pipeline : cfg_.pipelines) {
        for (const auto& node : pipeline.nodes) {
            const auto* model_id = inferModelId(node);
            if (!model_id) continue;
            if (*model_id == id) {
                references.pipelines[pipeline.id].direct_nodes.push_back(node.id);
            } else if (std::find(references.cascade_parents.begin(), references.cascade_parents.end(),
                                 *model_id) != references.cascade_parents.end()) {
                references.pipelines[pipeline.id].cascade_nodes.push_back(node.id);
            }
        }
    }
    return references;
}

ModelDetail TaskManager::describeModel(const std::string& id) const {
    ModelConfig repository_model;
    ModelDetail detail;
    detail.repository = describeRepositoryModelExport(cfg_.server.model_repository, id, &repository_model);
    bool has_repository_model = detail.repository.code.empty();
    if (detail.repository.code == "model_not_exportable") {
        try {
            repository_model = loadRepositoryModel(std::filesystem::path(cfg_.server.model_repository) / id);
            has_repository_model = true;
        } catch (const std::exception&) {
            // A model can still be registered even when its repository entry is unreadable.
        }
    }
    std::lock_guard lock(mu_);
    if (const auto* model = cfg_.findModel(id)) {
        detail.model = toModelInfo(*model);
        detail.registered = true;
        detail.cascade_children = model->cascade;
    } else if (has_repository_model) {
        detail.model = toModelInfo(repository_model);
        detail.cascade_children = repository_model.cascade;
    }
    if (!detail.model) return detail;

    detail.references = modelReferences(id);
    for (const auto& task : cfg_.tasks) {
        if (detail.references.pipelines.count(task.pipeline_id) == 0) continue;
        const auto entry = entries_.find(task.id);
        const auto state = entry == entries_.end() ? State::Stopped : entry->second.state;
        detail.tasks_by_pipeline[task.pipeline_id].emplace_back(task.id, state);
    }
    return detail;
}

nlohmann::json TaskManager::testModel(const std::string& id, const std::string& image) {
    const auto fail = [](const char* code, const char* message) {
        return nlohmann::json{{"status", "error"}, {"code", code}, {"message", message}};
    };
    if (!model_test_) return fail("model_test_unavailable", "model testing is unavailable");

    ModelConfig model;
    {
        std::lock_guard lock(mu_);
        const auto* registered = cfg_.findModel(id);
        if (!registered) return fail("model_not_registered", "register the model before testing");
        model = *registered;
    }
    std::unique_lock check_lock(model_check_mu_, std::try_to_lock);
    if (!check_lock.owns_lock()) return fail("model_test_busy", "another model is being checked");
    return model_test_(model, image);
}

bool TaskManager::referencesUnregisteredModel(const PipelineConfig& pipeline) const {
    const auto& removed_ids = runtime_state_.removed_model_ids;
    return std::any_of(pipeline.nodes.begin(), pipeline.nodes.end(), [&](const auto& node) {
        const auto* id = inferModelId(node);
        return id && !cfg_.findModel(*id) &&
               std::find(removed_ids.begin(), removed_ids.end(), *id) != removed_ids.end();
    });
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
    trackSourceProbeLocked(src);
    source_probe_cv_.notify_all();
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
    source_probes_.erase(id);
    source_probe_cv_.notify_all();
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
    if (cfg_.findPipeline(pipeline.id) || referencesUnregisteredModel(pipeline)) return false;
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
    if (!old_ptr || referencesUnregisteredModel(pipeline)) return false;

    for (const auto& task : cfg_.tasks) {
        if (task.pipeline_id != pipeline.id) continue;
        const auto it = entries_.find(task.id);
        if (it != entries_.end() && it->second.run) return false;
        auto runtime = pipeline;
        interpolatePipelineForTask(runtime, task, pipeline.id);
    }
    for (auto& p : cfg_.pipelines)
        if (p.id == pipeline.id) { p = pipeline; break; }
    for (auto& p : runtime_state_.added_pipelines)
        if (p.id == pipeline.id) { p = pipeline; break; }

    persist();
    return true;
}

ITaskManager::ActionResult TaskManager::requestUpdate(const TaskConfig& task) {
    std::lock_guard lock(mu_);
    const auto it = entries_.find(task.id);
    if (it == entries_.end() || it->second.removed) return ActionResult::NotFound;
    if (it->second.run) return ActionResult::Busy;
    if (!validateTask(task)) return ActionResult::NotFound;
    for (auto& t : cfg_.tasks)
        if (t.id == task.id) { t = task; break; }
    for (auto& t : runtime_state_.added_tasks)
        if (t.id == task.id) { t = task; break; }
    it->second.state = State::Stopped;
    it->second.failure.reset();
    persist();
    return ActionResult::Accepted;
}

bool TaskManager::updateTask(const TaskConfig& task) {
    return requestUpdate(task) == ActionResult::Accepted;
}

bool TaskManager::addTask(const TaskConfig& task, bool autostart) {
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.count(task.id)) return false;
    if (!validateTask(task)) return false;
    entries_.try_emplace(task.id);
    cfg_.tasks.push_back(task);
    auto& removed = runtime_state_.removed_task_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), task.id), removed.end());
    runtime_state_.added_tasks.push_back(task);
    if (autostart) forgetStoppedTask(runtime_state_, task.id);
    else rememberStoppedTask(runtime_state_, task.id);
    persist();
    if (autostart) startLocked(task.id);
    return true;
}

bool TaskManager::removeTask(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(id);
    if (it == entries_.end() || it->second.removed) return false;
    if (it->second.run) {
        it->second.removed = true;
        it->second.run->cancelled.store(true);
        lifecycle_cv_.notify_all();
    } else {
        entries_.erase(it);
    }
    cfg_.tasks.erase(std::remove_if(cfg_.tasks.begin(), cfg_.tasks.end(),
        [&](const auto& t){ return t.id == id; }), cfg_.tasks.end());
    auto& added = runtime_state_.added_tasks;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& t){ return t.id == id; }), added.end());
    runtime_state_.removed_task_ids.push_back(id);
    forgetStoppedTask(runtime_state_, id);
    persist();
    return true;
}

bool TaskManager::loadModel(const ModelConfig& model) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cfg_.findModel(model.id) ||
        std::any_of(model.cascade.begin(), model.cascade.end(),
                    [&](const auto& child) { return !cfg_.findModel(child.model_id); })) return false;
    cfg_.models.push_back(model);
    auto& removed = runtime_state_.removed_model_ids;
    removed.erase(std::remove(removed.begin(), removed.end(), model.id), removed.end());
    runtime_state_.added_models.push_back(model);
    persist();
    return true;
}

TaskManager::UnregisterResult TaskManager::unloadModel(const std::string& id) {
    std::lock_guard lock(mu_);
    if (!cfg_.findModel(id)) return UnregisterResult::NotFound;
    if (!modelReferences(id).empty()) return UnregisterResult::InUse;
    cfg_.models.erase(std::remove_if(cfg_.models.begin(), cfg_.models.end(),
        [&](const auto& m){ return m.id == id; }), cfg_.models.end());
    auto& added = runtime_state_.added_models;
    added.erase(std::remove_if(added.begin(), added.end(),
        [&](const auto& m){ return m.id == id; }), added.end());
    runtime_state_.removed_model_ids.push_back(id);
    persist();
    return UnregisterResult::Removed;
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

RepositoryModelExport TaskManager::describeModelExport(const std::string& id) const {
    return describeRepositoryModelExport(cfg_.server.model_repository, id);
}

ModelPublishResult TaskManager::publishModel(const std::string& id, const std::string& token,
                                            const std::string& attempt) {
    namespace fs = std::filesystem;
    static const std::regex id_pattern("^[A-Za-z0-9_-]+$");
    static const std::regex token_pattern("^[0-9a-f]{32}$");
    if (!std::regex_match(id, id_pattern) || !std::regex_match(token, token_pattern) ||
        (!attempt.empty() && !std::regex_match(attempt, token_pattern))) {
        return {"invalid_upload", "invalid model ID or upload token"};
    }
    if (cfg_.server.model_repository.empty() || !model_probe_) {
        return {"publish_unavailable", "model publication is unavailable"};
    }
    std::unique_lock publication_lock(model_check_mu_, std::try_to_lock);
    if (!publication_lock.owns_lock()) {
        return {"publish_busy", "another model is being checked, retry shortly"};
    }

    try {
        const auto root = fs::canonical(cfg_.server.model_repository);
        const auto target = root / id;
        const auto draft = root / ".uploads" / token;
        const auto receipt = draft / ".published-id";
        auto readLine = [](const fs::path& path) {
            std::ifstream input(path);
            std::string value;
            std::getline(input, value);
            return value;
        };
        if (readLine(target / ".publication-token") == token) return {};
        const auto previous_id = readLine(receipt);
        if (!previous_id.empty() && previous_id != id &&
            std::regex_match(previous_id, id_pattern) &&
            readLine(root / previous_id / ".publication-token") == token) {
            return {"already_published", "this upload was already published as " + previous_id};
        }

        auto checkId = [&]() -> ModelPublishResult {
            const auto& removed = runtime_state_.removed_model_ids;
            if (fs::symlink_status(target).type() != fs::file_type::not_found ||
                cfg_.findModel(id) || std::find(removed.begin(), removed.end(), id) != removed.end()) {
                return {"already_exists", "model ID already exists: " + id};
            }
            return {};
        };
        {
            std::lock_guard lock(mu_);
            auto result = checkId();
            if (!result.code.empty()) return result;
        }
        if (attempt.empty()) return {"upload_expired", "upload expired, select the model files again"};
        const auto candidate = draft / attempt / id;
        if (!fs::is_directory(candidate) || fs::canonical(candidate) != candidate) {
            return {"upload_missing", "engine cannot read the upload. Check the shared model repository"};
        }

        ModelConfig model;
        try {
            model = loadRepositoryModel(candidate);
            auto moveWeightPath = [&](std::string& weight) {
                if (weight.empty()) return;
                const auto path = fs::canonical(weight);
                const auto rel = path.lexically_relative(candidate);
                if (rel.empty() || *rel.begin() != "1" || !fs::is_regular_file(path) ||
                    fs::path(weight) != path) {
                    throw std::runtime_error("weight must be a regular file inside the uploaded version");
                }
                weight = (target / rel).string();
            };
            moveWeightPath(model.engine_path);
            moveWeightPath(model.onnx_path);
            for (auto& [_, weight] : model.om_paths) moveWeightPath(weight);
            if (model.input_shape.height < 1 || model.input_shape.width < 1 ||
                model.instance_count < 1 || model.batch_size < 1 || model.num_classes < 1) {
                throw std::runtime_error("input dimensions, batch size, classes and instance count must be positive");
            }
        } catch (const std::exception& e) {
            return {"invalid_model", e.what()};
        }

        auto checkRegisteredDependencies = [&]() -> ModelPublishResult {
            for (const auto& item : model.cascade) {
                if (!cfg_.findModel(item.model_id)) {
                    return {"missing_dependency", "register cascade model first: " + item.model_id};
                }
            }
            return {};
        };
        {
            std::lock_guard lock(mu_);
            auto result = checkRegisteredDependencies();
            if (!result.code.empty()) return result;
        }

        auto checkStartupConfig = [&]() -> ModelPublishResult {
            try {
                const auto startup = loadConfig(config_path_);
                if (startup.findModel(id)) {
                    return {"already_exists", "model ID already exists: " + id};
                }
                for (const auto& item : model.cascade) {
                    if (!startup.findModel(item.model_id)) {
                        return {"missing_dependency", "add cascade model to the startup config or model repository: " +
                                                      item.model_id};
                    }
                }
            } catch (const std::exception& e) {
                return {"startup_config_invalid", "cannot validate publication against startup config: " +
                                                  std::string(e.what())};
            }
            return {};
        };
        auto result = checkStartupConfig();
        if (!result.code.empty()) return result;

        try {
            model_probe_(candidate);
        } catch (const std::exception& e) {
            return {"weight_check_failed", e.what()};
        }

        result = checkStartupConfig();
        if (!result.code.empty()) return result;

        std::lock_guard lock(mu_);
        result = checkId();
        if (!result.code.empty()) return result;
        result = checkRegisteredDependencies();
        if (!result.code.empty()) return result;

        cfg_.models.reserve(cfg_.models.size() + 1);
        auto writeLine = [](const fs::path& path, const std::string& value) {
            std::ofstream out;
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.open(path);
            out << value << '\n';
            out.close();
        };
        writeLine(candidate / ".publication-token", token);
        writeLine(receipt, id);
        fs::rename(candidate, target);
        // The repository is the durable source for new publications. A restart
        // after the rename discovers the same checked model at its final path.
        cfg_.models.push_back(std::move(model));
        return {};
    } catch (const std::exception& e) {
        return {"publish_io_error", e.what()};
    }
}

} // namespace infer
