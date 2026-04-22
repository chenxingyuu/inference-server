#include "pipeline/InferWorkerGroup.h"
#include "common/Logger.h"

namespace infer {

InferWorkerGroup::InferWorkerGroup(const ModelConfig&  model_cfg,
                                   IPublisher&         publisher,
                                   BackendFactory      backend_factory,
                                   DecoderFactory      decoder_factory,
                                   std::shared_ptr<FrameArchiver> frame_archiver,
                                   std::shared_ptr<TrackerManager> tracker_manager,
                                   std::function<TrackerType(const std::string&)> tracker_type_resolver,
                                   std::function<ByteTrackConfig(const std::string&)> bytetrack_config_resolver) {
    const int count = std::max(1, model_cfg.instance_count);

    workers_.reserve(count);
    for (int i = 0; i < count; ++i) {
        // Determine device_id for this instance:
        // Use device_ids[i] if provided, otherwise fall back to model_cfg.device_id.
        ModelConfig instance_cfg = model_cfg;
        if (i < static_cast<int>(model_cfg.device_ids.size())) {
            instance_cfg.device_id = model_cfg.device_ids[i];
        }

        auto backend = backend_factory(instance_cfg);
        auto decoder = decoder_factory(instance_cfg);

        workers_.push_back(std::make_unique<InferWorker>(
            instance_cfg,
            std::move(backend),
            std::move(decoder),
            publisher,
            frame_archiver,
            tracker_manager,
            tracker_type_resolver,
            bytetrack_config_resolver));

        LOG_INFO("InferWorkerGroup [{}]: instance {} → device {}",
                 model_cfg.id, i, instance_cfg.device_id);
    }
}

void InferWorkerGroup::start() {
    for (auto& w : workers_) w->start();
}

void InferWorkerGroup::stop() {
    for (auto& w : workers_) w->stop();
}

void InferWorkerGroup::enqueue(Batch batch) {
    const std::size_t n = workers_.size();
    // Start from the round-robin candidate; skip RECOVERING/STOPPED workers.
    const std::size_t start = round_robin_idx_.fetch_add(1, std::memory_order_relaxed) % n;
    for (std::size_t i = 0; i < n; ++i) {
        auto& w = workers_[(start + i) % n];
        if (w->state() == WorkerState::RUNNING) {
            w->enqueue(std::move(batch));
            return;
        }
    }
    // No RUNNING worker found — fall back to first RECOVERING worker so the batch
    // is processed once recovery completes, rather than being dropped immediately.
    for (std::size_t i = 0; i < n; ++i) {
        auto& w = workers_[(start + i) % n];
        if (w->state() == WorkerState::RECOVERING) {
            w->enqueue(std::move(batch));
            return;
        }
    }
    // All workers STOPPED — drop
    LOG_ERROR("InferWorkerGroup: all workers stopped, dropping batch");
}

uint64_t InferWorkerGroup::processedBatches() const {
    uint64_t total = 0;
    for (const auto& w : workers_) total += w->processedBatches();
    return total;
}

uint64_t InferWorkerGroup::droppedBatches() const {
    uint64_t total = 0;
    for (const auto& w : workers_) total += w->droppedBatches();
    return total;
}

void InferWorkerGroup::setCascadeRouter(CascadeRouter* router, ResultMerger* merger) {
    for (auto& w : workers_) {
        w->setCascadeRouter(router, merger);
    }
}

} // namespace infer
