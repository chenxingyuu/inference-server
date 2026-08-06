#pragma once

#include "common/Config.h"
#include "pipeline/Event.h"
#include "pipeline/IStage.h"
#include "pipeline/InferWorkerGroup.h"
#include "publisher/IPublisher.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace infer {

// DAG infer.engine stage: batches frames like InferEngineStage, then dispatches
// Batches to InferWorkerGroup so models.instance_count and models.device_ids apply.
class InferEngineWorkerStage final : public IStage {
public:
    using BackendFactory = InferWorkerGroup::BackendFactory;
    using DecoderFactory = InferWorkerGroup::DecoderFactory;

    InferEngineWorkerStage(std::string id,
                           ModelConfig model_cfg,
                           BackendFactory backend_factory,
                           DecoderFactory decoder_factory);

    ~InferEngineWorkerStage() override;

    std::string id() const override;

    void start() override;
    void stop() override;
    void onGraphExecutorDraining() noexcept override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

    // Test hook: InferWorkerGroup instance count after construction.
    int workerInstanceCount() const { return group_ ? group_->instanceCount() : 0; }

    // Test hook: number of results still awaited from the worker group.
    std::size_t inflightSize() const;

    // Test hook: shorten the stale-inflight eviction timeout for deterministic tests.
    void setInflightStaleTimeoutForTest(std::chrono::milliseconds ms);

private:
    struct Relay final : IPublisher {
        InferEngineWorkerStage* owner{nullptr};
        Relay() = default;
        void bind(InferEngineWorkerStage* o) { owner = o; }
        void publish(InferResult r) override;
        void flush() override {}
    };

    struct PendingEmit {
        EventEnvelope envelope;
        EmitFn        emit;
        // Time the frame was handed to the worker group. Used to evict entries
        // whose inference result never arrives (dropped batch / fault / exception).
        std::chrono::steady_clock::time_point submitted_at{};
    };

    using PendingKey = std::pair<std::string, uint64_t>;

    // Frames with no result after this long are treated as dropped and evicted,
    // releasing the shared_ptr<Frame> they pin. Comfortably above worst-case
    // (queue wait + inference) latency so in-flight frames are never evicted early.
    static constexpr int64_t kInflightStaleMs = 5000;

    std::vector<PendingEmit>   extractBatch(int flush_count);
    void                       flushToWorkers(std::vector<PendingEmit> events);
    void                       flushLoop();
    std::size_t                pendingQueueSize();
    void                       sweepStaleInflightLocked();  // caller holds inflight_mutex_

    void onInferResult(InferResult r);

    int effectiveBatchSize() const;

    std::string id_;
    ModelConfig model_cfg_;

    BackendFactory backend_factory_;
    DecoderFactory decoder_factory_;

    Relay relay_{};
    std::unique_ptr<InferWorkerGroup> group_;

    std::mutex                pending_mutex_;
    std::deque<PendingEmit>   pending_events_;   // each entry owns its emit fn
    std::chrono::steady_clock::time_point batch_deadline_;

    mutable std::mutex                inflight_mutex_;
    std::map<PendingKey, PendingEmit> inflight_;
    std::chrono::milliseconds         inflight_stale_{kInflightStaleMs};  // guarded by inflight_mutex_

    std::thread       flush_thread_;
    std::atomic<bool> flush_stop_{false};
    std::condition_variable flush_cv_;

    int  max_pending_{0};
    bool started_{false};
    std::atomic<bool> draining_{false};
};

} // namespace infer
