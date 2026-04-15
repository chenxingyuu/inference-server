#include "pipeline/ModelManager.h"
#include "common/Logger.h"
#include <stdexcept>
#include <thread>
#include <chrono>

namespace infer {

ModelManager::ModelManager(StreamPool&                     pool,
                           IPublisher&                     publisher,
                           BackendFactory                  backend_factory,
                           DecoderFactory                  decoder_factory,
                           const std::vector<ModelConfig>& all_model_configs)
    : pool_(pool)
    , publisher_(publisher)
    , backend_factory_(std::move(backend_factory))
    , decoder_factory_(std::move(decoder_factory))
{
    for (const auto& mc : all_model_configs) {
        all_model_cfgs_[mc.id] = mc;
    }
}

void ModelManager::load(const ModelConfig& cfg) {
    std::lock_guard<std::mutex> lk(mu_);

    // Register config for cascade secondary lookup (enables REST-loaded secondaries)
    all_model_cfgs_[cfg.id] = cfg;

    auto it = entries_.find(cfg.id);
    if (it != entries_.end() && it->second.state == State::READY) {
        LOG_WARN("ModelManager: model {} already loaded, ignoring load()", cfg.id);
        return;
    }

    auto& entry = entries_[cfg.id];
    entry.cfg   = cfg;
    entry.state = State::LOADING;

    try {
        startPipeline(entry);
        entry.state = State::READY;
        LOG_INFO("ModelManager: model {} loaded", cfg.id);
    } catch (const std::exception& e) {
        entries_.erase(cfg.id);
        throw std::runtime_error("ModelManager: load failed for " + cfg.id
                                  + ": " + e.what());
    }
}

void ModelManager::unload(const std::string& model_id) {
    std::unique_lock<std::mutex> lk(mu_);

    auto it = entries_.find(model_id);
    if (it == entries_.end()) {
        LOG_WARN("ModelManager: model {} not found, ignoring unload()", model_id);
        return;
    }

    auto& entry = it->second;
    entry.state = State::DRAINING;

    auto group            = std::move(entry.group);
    auto scheduler        = std::move(entry.scheduler);
    auto secondary_groups = std::move(entry.secondary_groups);
    CascadeOwnership cascade;
    cascade.merger        = std::move(entry.merger);
    cascade.router        = std::move(entry.router);
    cascade.attr_publishers = std::move(entry.attr_publishers);
    entries_.erase(model_id);
    lk.unlock();

    stopPipeline_unlocked(std::move(scheduler), std::move(group),
                          std::move(secondary_groups), std::move(cascade), model_id);
}

void ModelManager::swap(const std::string& model_id, const ModelConfig& new_cfg) {
    std::unique_lock<std::mutex> lk(mu_);

    auto it = entries_.find(model_id);
    if (it == entries_.end()) {
        lk.unlock();
        load(new_cfg);
        return;
    }

    ModelEntry new_entry;
    new_entry.cfg   = new_cfg;
    new_entry.state = State::LOADING;

    try {
        startPipeline(new_entry);
        new_entry.state = State::READY;
    } catch (const std::exception& e) {
        throw std::runtime_error("ModelManager: swap load failed for "
                                  + new_cfg.id + ": " + e.what());
    }

    auto old_group            = std::move(it->second.group);
    auto old_scheduler        = std::move(it->second.scheduler);
    auto old_secondary_groups = std::move(it->second.secondary_groups);
    CascadeOwnership old_cascade;
    old_cascade.merger        = std::move(it->second.merger);
    old_cascade.router        = std::move(it->second.router);
    old_cascade.attr_publishers = std::move(it->second.attr_publishers);
    it->second = std::move(new_entry);
    lk.unlock();

    stopPipeline_unlocked(std::move(old_scheduler), std::move(old_group),
                          std::move(old_secondary_groups), std::move(old_cascade), model_id);
    LOG_INFO("ModelManager: model {} swapped to new config", model_id);
}

std::vector<std::pair<std::string, ModelManager::State>>
ModelManager::listModels() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::pair<std::string, State>> result;
    result.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        result.emplace_back(id, entry.state);
    }
    return result;
}

const ModelManager::ModelEntry* ModelManager::find(const std::string& model_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = entries_.find(model_id);
    return (it != entries_.end()) ? &it->second : nullptr;
}

void ModelManager::startPipeline(ModelEntry& entry) {
    if (entry.cfg.model_type == ModelType::Detector) {
        entry.tracker_manager = std::make_shared<TrackerManager>();
    }

    // Create primary InferWorkerGroup (uses Kafka publisher by default)
    entry.group = std::make_unique<InferWorkerGroup>(
        entry.cfg,
        publisher_,
        backend_factory_,
        decoder_factory_,
        entry.tracker_manager,
        [this](const std::string& stream_id) { return pool_.getStreamTrackerType(stream_id); });

    // Wire cascade if this is a primary model with secondary classifiers
    if (!entry.cfg.cascade.empty()) {
        entry.merger = std::make_unique<ResultMerger>(publisher_);

        std::unordered_map<std::string, InferWorkerGroup*> sec_map;

        for (const auto& cas : entry.cfg.cascade) {
            // Look up the secondary model config
            auto cfg_it = all_model_cfgs_.find(cas.model_id);
            if (cfg_it == all_model_cfgs_.end()) {
                LOG_WARN("ModelManager: cascade secondary config '{}' not found — skipping",
                         cas.model_id);
                continue;
            }

            // Create AttributePublisher that routes to this primary's merger
            auto attr_pub = std::make_unique<AttributePublisher>(
                *entry.merger, cas.attribute_key);
            IPublisher& attr_pub_ref = *attr_pub;
            entry.attr_publishers.push_back(std::move(attr_pub));

            // Create secondary InferWorkerGroup with AttributePublisher
            auto sec_group = std::make_unique<InferWorkerGroup>(
                cfg_it->second,
                attr_pub_ref,
                backend_factory_,
                decoder_factory_,
                nullptr,
                [](const std::string&) { return TrackerType::None; });
            sec_map[cas.model_id] = sec_group.get();
            entry.secondary_groups.push_back(std::move(sec_group));
        }

        if (!sec_map.empty()) {
            entry.router = std::make_unique<CascadeRouter>(
                entry.cfg, std::move(sec_map), *entry.merger);
            entry.group->setCascadeRouter(entry.router.get(), entry.merger.get());

            // Start secondary workers before the primary
            for (auto& sg : entry.secondary_groups) sg->start();
        }
    }

    // Create BatchScheduler pointing to the primary group
    InferWorkerGroup* group_ptr = entry.group.get();
    entry.scheduler = std::make_unique<BatchScheduler>(
        entry.cfg, pool_,
        [group_ptr](Batch b) { group_ptr->enqueue(std::move(b)); });

    entry.group->start();
    entry.scheduler->start();
}

void ModelManager::stopPipeline_unlocked(
        std::unique_ptr<BatchScheduler>                 scheduler,
        std::unique_ptr<InferWorkerGroup>               group,
        std::vector<std::unique_ptr<InferWorkerGroup>>  secondary_groups,
        CascadeOwnership                                cascade,
        const std::string& model_id) {
    // Stop scheduler first: no new batches will be sent to the primary group.
    if (scheduler) scheduler->stop();

    // Brief drain: give in-flight batches a chance to complete.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // Stop primary workers.
    if (group) group->stop();

    // Stop secondary workers before destroying cascade objects.
    // Secondary workers call AttributePublisher::publish() → ResultMerger::addAttribute(),
    // so cascade.merger and cascade.attr_publishers must outlive these threads.
    for (auto& sg : secondary_groups) sg->stop();

    // cascade (merger, router, attr_publishers) is destroyed here, safely after
    // all secondary workers have halted.
    LOG_INFO("ModelManager: model {} unloaded", model_id);
}

} // namespace infer
