#include "pipeline/BatchScheduler.h"
#include "common/Logger.h"
#include <chrono>

namespace infer {

BatchScheduler::BatchScheduler(const ModelConfig& model_cfg,
                                StreamPool&        pool,
                                BatchCallback      callback)
    : model_cfg_(model_cfg)
    , pool_(pool)
    , callback_(std::move(callback))
{}

BatchScheduler::~BatchScheduler() {
    stop();
}

void BatchScheduler::start() {
    if (running_.load()) return;
    stop_flag_ = false;
    thread_    = std::thread(&BatchScheduler::scheduleLoop, this);
}

void BatchScheduler::stop() {
    stop_flag_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

void BatchScheduler::scheduleLoop() {
    running_.store(true);
    LOG_INFO("BatchScheduler[{}]: started", model_cfg_.id);

    const int max_bs   = model_cfg_.batch_size;
    Batch     batch;
    batch.frames.reserve(max_bs);
    batch.gpu_frames.reserve(max_bs);
    batch.metas.reserve(max_bs);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(kMaxWaitMs);

    while (!stop_flag_.load()) {
        // Collect frames from streams assigned to this model
        auto streams = pool_.activeStreams();
        for (const auto& sid : streams) {
            if (pool_.getStreamModelId(sid) != model_cfg_.id) continue;

            FrameBuffer* buf = pool_.getBuffer(sid);
            if (!buf) continue;

            Frame f;
            if (buf->pop(f)) {
                if (f.meta.model_id.empty()) f.meta.model_id = model_cfg_.id;

                // Enforce a homogeneous batch: once the type is set, reject
                // frames of the opposite type to prevent OOB in the backend.
                if (!batch.empty() && f.is_gpu != batch.is_gpu) {
                    LOG_WARN("BatchScheduler[{}]: mixed GPU/CPU frames in one batch, "
                             "dropping frame from stream {}", model_cfg_.id, sid);
                    continue;
                }

                if (f.is_gpu) {
                    batch.gpu_frames.push_back(std::move(f.gpu_buf));
                    batch.is_gpu = true;
                } else {
                    batch.frames.push_back(std::move(f.image));
                }
                batch.metas.push_back(std::move(f.meta));
            }

            if (batch.size() >= max_bs) break;
        }

        auto now = std::chrono::steady_clock::now();
        bool timeout = (now >= deadline);

        if (!batch.empty() && (batch.size() >= max_bs || timeout)) {
            callback_(std::move(batch));
            batch.frames.clear();
            batch.gpu_frames.clear();
            batch.metas.clear();
            batch.is_gpu = false;
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(kMaxWaitMs);
        } else if (batch.empty()) {
            // Nothing to do; yield briefly
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            if (timeout) {
                deadline = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(kMaxWaitMs);
            }
        }
    }

    LOG_INFO("BatchScheduler[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

} // namespace infer
