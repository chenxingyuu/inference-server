#include "pipeline/PipelineManager.h"
#include "pipeline/StageFactory.h"

namespace infer {

PipelineManager::PipelineManager(const AppConfig& cfg,
                                 IPublisher& publisher,
                                 std::shared_ptr<FrameArchiver> frame_archiver)
    : cfg_(cfg), publisher_(publisher), frame_archiver_(std::move(frame_archiver)) {}

void PipelineManager::loadAll() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
    for (const auto& pipeline_cfg : cfg_.pipelines) {
        auto source = cfg_.findSource(pipeline_cfg.source_id);
        if (!source) continue;
        auto executor = std::make_unique<GraphExecutor>(pipeline_cfg);
        StageFactory::Context ctx{cfg_, *source, publisher_, frame_archiver_};
        for (const auto& node : pipeline_cfg.nodes) {
            executor->addStage(StageFactory::create(node, ctx));
        }
        executor->build();
        entries_[pipeline_cfg.id] = Entry{std::move(executor), State::Stopped};
    }
}

void PipelineManager::startAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Running) continue;
        entry.executor->start();
        entry.state = State::Running;
    }
}

void PipelineManager::stopAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, entry] : entries_) {
        if (entry.state == State::Stopped) continue;
        entry.executor->stop();
        entry.state = State::Stopped;
    }
}

bool PipelineManager::start(const std::string& pipeline_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(pipeline_id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Stopped) {
        it->second.executor->start();
        it->second.state = State::Running;
    }
    return true;
}

bool PipelineManager::stop(const std::string& pipeline_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find(pipeline_id);
    if (it == entries_.end()) return false;
    if (it->second.state == State::Running) {
        it->second.executor->stop();
        it->second.state = State::Stopped;
    }
    return true;
}

std::vector<std::pair<std::string, PipelineManager::State>> PipelineManager::listPipelines() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<std::string, State>> out;
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) out.emplace_back(id, entry.state);
    return out;
}

} // namespace infer
