#pragma once

#include "infer/IInferBackend.h"
#include "infer/GpuFault.h"
#include "decoder/IYOLODecoder.h"
#include "publisher/IPublisher.h"
#include "common/Config.h"
#include "pipeline/BatchPolicy.h"
#include "tracker/TrackerManager.h"
#include "archive/FrameArchiver.h"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace infer { class CascadeRouter; class ResultMerger; }

namespace infer {

enum class WorkerState {
    RUNNING    = 0,
    RECOVERING = 1,
    STOPPED    = 2,
};

// Single-device inference worker with GPU fault self-healing.
// Receives Batches from BatchScheduler, runs inference, decodes output,
// and pushes InferResult objects to IPublisher.
//
// Fault handling:
//   GpuMemoryPressureException → back-pressure: reduce batch tier, re-queue, sleep 50ms
//   GpuFaultException(OOM)     → same as pressure (engine still alive)
//   GpuFaultException(other)   → recoverFromFault(): unload + reload engine (up to 3 retries)
//   All retries fail            → state = STOPPED, critical log emitted
class InferWorker {
public:
    InferWorker(const ModelConfig&           model_cfg,
                std::unique_ptr<IInferBackend> backend,
                std::unique_ptr<IYOLODecoder>  decoder,
                IPublisher&                    publisher,
                std::shared_ptr<FrameArchiver> frame_archiver,
                std::shared_ptr<TrackerManager> tracker_manager,
                std::function<TrackerType(const std::string&)> tracker_type_resolver,
                std::function<ByteTrackConfig(const std::string&)> bytetrack_config_resolver);
    ~InferWorker();

    void start();
    void stop();
    bool running() const { return running_.load(); }

    // Thread-safe enqueue — called by BatchScheduler.
    // Drops batch silently if queue is full or worker is STOPPED.
    void enqueue(Batch batch);

    WorkerState state() const { return state_.load(); }
    int cudaDeviceId()   const { return model_cfg_.device_id; }
    int ascendDeviceId() const { return model_cfg_.device_id; }

    // Optional: set a cascade router for primary→secondary routing (Phase 7b-1).
    // Must be called before start(). Not thread-safe.
    void setCascadeRouter(CascadeRouter* router, ResultMerger* merger) {
        cascade_router_ = router;
        result_merger_  = merger;
    }

    uint64_t processedBatches()  const { return processed_batches_.load(); }
    uint64_t droppedBatches()    const { return dropped_batches_.load(); }

private:
    void workerLoop();
    void recoverFromFault(GpuFaultType fault);
    void enqueueHead(Batch batch); // re-insert at front of queue (OOM back-pressure)

    ModelConfig                    model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder>  decoder_;
    IPublisher&                    publisher_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    std::shared_ptr<TrackerManager> tracker_manager_;
    std::function<TrackerType(const std::string&)> tracker_type_resolver_;
    std::function<ByteTrackConfig(const std::string&)> bytetrack_config_resolver_;

    std::deque<Batch>       queue_; // deque allows front-insertion for OOM re-queue
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::atomic<bool>       stop_flag_{false};
    std::atomic<WorkerState> state_{WorkerState::RUNNING};

    BatchPolicy           batch_policy_;
    std::atomic<uint64_t> processed_batches_{0};
    std::atomic<uint64_t> dropped_batches_{0};

    // Phase 7b-1: optional cascade routing
    CascadeRouter* cascade_router_{nullptr};
    ResultMerger*  result_merger_{nullptr};

    static constexpr std::size_t kMaxQueueSize = 32;
};

} // namespace infer
