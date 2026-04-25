#include "pipeline/TaskManager.h"
#include "pipeline/StageFactory.h"
#include "stream/GpuDeviceAllocator.h"
#include "common/Logger.h"

#include <set>

namespace infer {

TaskManager::TaskManager(const AppConfig& cfg,
                         IPublisher& publisher,
                         std::shared_ptr<FrameArchiver> frame_archiver)
    : cfg_(cfg), publisher_(publisher), frame_archiver_(std::move(frame_archiver)) {}

void TaskManager::loadAll() {
    std::lock_guard<std::mutex> lock(mu_);
    // Stop running executors before destroying them so their worker threads are
    // joined cleanly and no thread is inside stage->process() during teardown.
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Running) {
            entry.executor->stop();
            entry.state = State::Stopped;
        }
    }
    entries_.clear();

    // Build a shared GpuDeviceAllocator from all model device IDs so that
    // SourceRtspStage instances can automatically pick the least-loaded GPU.
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
        auto source = cfg_.findSource(task.source_id);
        auto pipeline_tpl = cfg_.findPipeline(task.pipeline_id);
        if (!source || !pipeline_tpl) continue;
        PipelineConfig runtime_cfg = *pipeline_tpl;
        runtime_cfg.id = task.id;
        auto executor = std::make_unique<GraphExecutor>(runtime_cfg);
        StageFactory::Context ctx{cfg_, *source, publisher_, frame_archiver_, task.sample_fps, task.sampling_mode, task.use_hwdec, gpu_alloc};
        for (const auto& node : runtime_cfg.nodes) {
            executor->addStage(StageFactory::create(node, ctx));
        }
        executor->build();
        LOG_INFO(
            "Pipeline built for task={} (template_pipeline={}, source={})\n{}",
            task.id,
            task.pipeline_id,
            task.source_id,
            executor->graphText());
        entries_[task.id] = Entry{std::move(executor), State::Stopped};
    }
}

void TaskManager::startAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Running) continue;
        entry.executor->start();
        entry.state = State::Running;
    }
}

void TaskManager::stopAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Stopped) continue;
        entry.executor->stop();
        entry.state = State::Stopped;
    }
}

bool TaskManager::start(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(task_id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Stopped) {
        it->second.executor->start();
        it->second.state = State::Running;
    }
    return true;
}

bool TaskManager::stop(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(task_id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Running) {
        it->second.executor->stop();
        it->second.state = State::Stopped;
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

} // namespace infer
