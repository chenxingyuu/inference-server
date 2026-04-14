#pragma once

#include "stream/StreamPool.h"
#include "common/Config.h"
#include <functional>
#include <thread>
#include <atomic>

namespace infer {

using BatchCallback = std::function<void(Batch)>;

// Polls all FrameBuffers for a given model, assembles batches, and calls the
// callback on the assembled Batch.
//
// Strategy:
//   - Iterate all streams assigned to this model
//   - Accumulate frames up to max_batch_size
//   - Flush when either max_batch_size is reached OR max_wait_ms has elapsed
class BatchScheduler {
public:
    BatchScheduler(const ModelConfig& model_cfg,
                   StreamPool&        pool,
                   BatchCallback      callback);
    ~BatchScheduler();

    void start();
    void stop();
    bool running() const { return running_.load(); }

private:
    void scheduleLoop();

    ModelConfig   model_cfg_;
    StreamPool&   pool_;
    BatchCallback callback_;

    std::thread        thread_;
    std::atomic<bool>  running_{false};
    std::atomic<bool>  stop_flag_{false};

    static constexpr int kMaxWaitMs = 10;
};

} // namespace infer
