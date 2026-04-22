#include "pipeline/BatchScheduler.h"
#include "common/Logger.h"
#include <chrono>
#include <algorithm>

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

    // Build preferred batch sizes list (Triton-style, Phase 7b).
    // Sort descending so we flush at the largest achievable preferred size.
    // If no preferred sizes are configured, use batch_size as the single target.
    std::vector<int> preferred = model_cfg_.preferred_batch_sizes;
    if (preferred.empty()) {
        preferred.push_back(model_cfg_.batch_size);
    } else {
        std::sort(preferred.begin(), preferred.end(), std::greater<int>());
    }
    const int max_bs = preferred.front();   // hard upper bound
    const int delay_us = model_cfg_.max_queue_delay_us;
    {
        std::string pref_str;
        for (int p : preferred) pref_str += std::to_string(p) + " ";
        LOG_DEBUG("BatchScheduler[{}]: preferred_sizes=[{}] max_bs={} delay_us={}",
                  model_cfg_.id, pref_str, max_bs, delay_us);
    }

    Batch batch;
    batch.frames.reserve(max_bs);
    batch.gpu_frames.reserve(max_bs);
    batch.metas.reserve(max_bs);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(delay_us);

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

            if (static_cast<int>(batch.size()) >= max_bs) break;
        }

        auto now     = std::chrono::steady_clock::now();
        bool timeout = (now >= deadline);

        // Flush decision: prefer largest target size to improve GPU utilisation.
        // A partial batch is sent if the deadline has passed.
        bool should_flush = false;
        if (!batch.empty()) {
            int bs = static_cast<int>(batch.size());
            // Check if we hit any preferred batch size (largest first)
            for (int target : preferred) {
                if (bs >= target) {
                    should_flush = true;
                    break;
                }
            }
            if (!should_flush && timeout) should_flush = true;
        }

        if (should_flush) {
            LOG_DEBUG("BatchScheduler[{}]: flush batch_size={} trigger={} active_streams={}",
                      model_cfg_.id, batch.size(), timeout ? "timeout" : "full",
                      pool_.activeStreams().size());
            callback_(std::move(batch));
            batch.frames.clear();
            batch.gpu_frames.clear();
            batch.metas.clear();
            batch.is_gpu = false;
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::microseconds(delay_us);
        } else if (batch.empty()) {
            // Nothing to do; yield briefly
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            if (timeout) {
                deadline = std::chrono::steady_clock::now() +
                           std::chrono::microseconds(delay_us);
            }
        }
    }

    LOG_INFO("BatchScheduler[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

} // namespace infer
