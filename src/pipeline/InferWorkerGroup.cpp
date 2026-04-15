#include "pipeline/InferWorkerGroup.h"
#include "common/Logger.h"

namespace infer {

InferWorkerGroup::InferWorkerGroup(const ModelConfig&  model_cfg,
                                   IPublisher&         publisher,
                                   BackendFactory      backend_factory,
                                   DecoderFactory      decoder_factory) {
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
            publisher));

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
    // Round-robin assignment: atomically pick the next worker index.
    const std::size_t n   = workers_.size();
    const std::size_t idx = round_robin_idx_.fetch_add(1, std::memory_order_relaxed) % n;
    workers_[idx]->enqueue(std::move(batch));
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
