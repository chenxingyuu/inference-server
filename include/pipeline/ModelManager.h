#pragma once

#include "pipeline/InferWorkerGroup.h"
#include "pipeline/BatchScheduler.h"
#include "pipeline/ResultMerger.h"
#include "pipeline/CascadeRouter.h"
#include "stream/StreamPool.h"
#include "publisher/IPublisher.h"
#include "publisher/AttributePublisher.h"
#include "common/Config.h"
#include "archive/FrameArchiver.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>

namespace infer {

// Manages the lifecycle of model pipelines at runtime.
// Inspired by Triton's model repository: models can be loaded, unloaded, and
// hot-swapped without restarting the server.
//
// Cascade wiring:
//   Primary models with cascade config have their secondary InferWorkerGroups
//   created internally (with AttributePublisher) and wired through CascadeRouter
//   + ResultMerger. Callers should NOT load cascade secondary models separately;
//   pass all_model_configs at construction so ModelManager can locate them.
//
// State machine per model:
//   UNLOADED → LOADING → READY → DRAINING → UNLOADING → UNLOADED
//                          ↑                                │
//                          └──────────── SWAP ──────────────┘
class ModelManager {
public:
    enum class State { UNLOADED, LOADING, READY, DRAINING, UNLOADING };

    struct ModelEntry {
        ModelConfig                          cfg;
        std::unique_ptr<InferWorkerGroup>    group;
        std::unique_ptr<BatchScheduler>      scheduler;
        State                                state{State::UNLOADED};

        // Cascade infrastructure (set for primary models with cascade config):
        std::unique_ptr<ResultMerger>                    merger;
        std::unique_ptr<CascadeRouter>                   router;
        std::vector<std::unique_ptr<InferWorkerGroup>>   secondary_groups;
        std::vector<std::unique_ptr<AttributePublisher>> attr_publishers;
        std::shared_ptr<TrackerManager>                  tracker_manager;
    };

    using BackendFactory = std::function<std::unique_ptr<IInferBackend>(const ModelConfig&)>;
    using DecoderFactory = std::function<std::unique_ptr<IYOLODecoder>(const ModelConfig&)>;

    // all_model_configs: used to look up secondary model configs for cascade wiring.
    ModelManager(StreamPool&                      pool,
                 IPublisher&                      publisher,
                 std::shared_ptr<FrameArchiver>   frame_archiver,
                 BackendFactory                   backend_factory,
                 DecoderFactory                   decoder_factory,
                 const std::vector<ModelConfig>&  all_model_configs = {});

    // Load a new model pipeline. Throws on error. No-op if already loaded.
    void load(const ModelConfig& cfg);

    // Drain and unload a model. Stops BatchScheduler first, then workers.
    // After this call the model is fully stopped and can be reloaded.
    void unload(const std::string& model_id);

    // Atomically replace an existing model with a new configuration.
    // Starts the new pipeline, stops the old one, then destroys it.
    // Brief overlap period (< ~30ms) where both pipelines exist.
    void swap(const std::string& model_id, const ModelConfig& new_cfg);

    // List all model IDs and their current state.
    std::vector<std::pair<std::string, State>> listModels() const;

    // Returns nullptr if model not found.
    const ModelEntry* find(const std::string& model_id) const;

private:
    void startPipeline(ModelEntry& entry);

    // Destroy pipeline components outside of mu_ to prevent deadlock.
    // Cascade objects (merger, router, attr_publishers) are passed in so they
    // outlive the secondary workers — workers may call AttributePublisher::publish()
    // right up to stop(), so they must not be destroyed before workers halt.
    struct CascadeOwnership {
        std::unique_ptr<ResultMerger>                    merger;
        std::unique_ptr<CascadeRouter>                   router;
        std::vector<std::unique_ptr<AttributePublisher>> attr_publishers;
    };
    void stopPipeline_unlocked(std::unique_ptr<BatchScheduler>                 scheduler,
                               std::unique_ptr<InferWorkerGroup>               group,
                               std::vector<std::unique_ptr<InferWorkerGroup>>  secondary_groups,
                               CascadeOwnership                                cascade,
                               const std::string&                              model_id);

    StreamPool&    pool_;
    IPublisher&    publisher_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    BackendFactory backend_factory_;
    DecoderFactory decoder_factory_;

    // Secondary model config lookup (by model id)
    std::unordered_map<std::string, ModelConfig> all_model_cfgs_;

    mutable std::mutex                           mu_;
    std::unordered_map<std::string, ModelEntry>  entries_;
};

} // namespace infer
