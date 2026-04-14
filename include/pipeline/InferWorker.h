#pragma once

#include "infer/IInferBackend.h"
#include "decoder/IYOLODecoder.h"
#include "publisher/IPublisher.h"
#include "common/Config.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace infer {

// Single-device inference worker.
// Receives Batches from BatchScheduler, runs inference, decodes output,
// and pushes InferResult objects to IPublisher.
class InferWorker {
public:
    InferWorker(const ModelConfig&           model_cfg,
                std::unique_ptr<IInferBackend> backend,
                std::unique_ptr<IYOLODecoder>  decoder,
                IPublisher&                    publisher);
    ~InferWorker();

    void start();
    void stop();
    bool running() const { return running_.load(); }

    // Thread-safe enqueue — called by BatchScheduler
    void enqueue(Batch batch);

    uint64_t processedBatches()  const { return processed_batches_.load(); }
    uint64_t droppedBatches()    const { return dropped_batches_.load(); }

private:
    void workerLoop();

    ModelConfig                    model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder>  decoder_;
    IPublisher&                    publisher_;

    std::queue<Batch>       queue_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::atomic<bool>       stop_flag_{false};

    std::atomic<uint64_t>   processed_batches_{0};
    std::atomic<uint64_t>   dropped_batches_{0};

    static constexpr std::size_t kMaxQueueSize = 32;
};

} // namespace infer
