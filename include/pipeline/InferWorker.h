#pragma once

#include "infer/IInferBackend.h"
#include "decoder/IYOLODecoder.h"
#include "publisher/IPublisher.h"
#include "common/Config.h"
#include "tracker/TrackerManager.h"
#include "archive/FrameArchiver.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace infer { class CascadeRouter; class ResultMerger; }

namespace infer {

// Single-device inference worker.
// Receives Batches from BatchScheduler, runs inference, decodes output,
// and pushes InferResult objects to IPublisher.
class InferWorker {
public:
    InferWorker(const ModelConfig&           model_cfg,
                std::unique_ptr<IInferBackend> backend,
                std::unique_ptr<IYOLODecoder>  decoder,
                IPublisher&                    publisher,
                std::shared_ptr<FrameArchiver> frame_archiver,
                std::shared_ptr<TrackerManager> tracker_manager,
                std::function<TrackerType(const std::string&)> tracker_type_resolver);
    ~InferWorker();

    void start();
    void stop();
    bool running() const { return running_.load(); }

    // Thread-safe enqueue — called by BatchScheduler
    void enqueue(Batch batch);

    // Optional: set a cascade router for primary→secondary routing (Phase 7b-1).
    // When set, results are routed through the router instead of published directly.
    // Must be called before start(). Not thread-safe.
    void setCascadeRouter(CascadeRouter* router, ResultMerger* merger) {
        cascade_router_ = router;
        result_merger_  = merger;
    }

    uint64_t processedBatches()  const { return processed_batches_.load(); }
    uint64_t droppedBatches()    const { return dropped_batches_.load(); }

private:
    void workerLoop();

    ModelConfig                    model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder>  decoder_;
    IPublisher&                    publisher_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    std::shared_ptr<TrackerManager> tracker_manager_;
    std::function<TrackerType(const std::string&)> tracker_type_resolver_;

    std::queue<Batch>       queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::atomic<bool>       stop_flag_{false};

    std::atomic<uint64_t>   processed_batches_{0};
    std::atomic<uint64_t>   dropped_batches_{0};

    // Phase 7b-1: optional cascade routing
    CascadeRouter* cascade_router_{nullptr};
    ResultMerger*  result_merger_{nullptr};

    static constexpr std::size_t kMaxQueueSize = 32;
};

} // namespace infer
