#pragma once

#include "pipeline/InferWorker.h"
#include "infer/IInferBackend.h"
#include "decoder/IYOLODecoder.h"
#include "publisher/IPublisher.h"
#include "common/Config.h"
#include <vector>
#include <memory>
#include <atomic>
#include <functional>

namespace infer { class CascadeRouter; class ResultMerger; }

namespace infer {

// Manages N parallel InferWorker instances for the same model.
// Inspired by Triton's instance_group: multiple execution contexts share
// one engine/model file but run on independent CUDA contexts/streams,
// enabling higher throughput on multi-GPU or multi-context setups.
//
// Batch distribution: round-robin across instances (lock-free atomic index).
// Each instance has its own queue, backend, and CUDA stream.
class InferWorkerGroup {
public:
    // Factory function type: creates a backend for the given per-instance device_id.
    using BackendFactory = std::function<std::unique_ptr<IInferBackend>(const ModelConfig&)>;
    using DecoderFactory = std::function<std::unique_ptr<IYOLODecoder>(const ModelConfig&)>;

    InferWorkerGroup(const ModelConfig&  model_cfg,
                     IPublisher&         publisher,
                     BackendFactory      backend_factory,
                     DecoderFactory      decoder_factory);
    ~InferWorkerGroup() = default;

    void start();
    void stop();

    // Distribute batch to the next available worker (round-robin).
    void enqueue(Batch batch);

    // Wire cascade routing on every worker. Must be called before start().
    void setCascadeRouter(CascadeRouter* router, ResultMerger* merger);

    int  instanceCount()       const { return static_cast<int>(workers_.size()); }
    uint64_t processedBatches() const;
    uint64_t droppedBatches()   const;

private:
    std::vector<std::unique_ptr<InferWorker>> workers_;
    std::atomic<uint64_t> round_robin_idx_{0};
};

} // namespace infer
