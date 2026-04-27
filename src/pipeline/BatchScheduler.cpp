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
#ifdef BUILD_ASCEND_BACKEND
    batch.ascend_frames.reserve(max_bs);
#endif

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(delay_us);
    auto stream_refresh_deadline = std::chrono::steady_clock::now();
    std::vector<std::string> model_streams;

    while (!stop_flag_.load()) {
        // Refresh stream list periodically to reduce lock pressure on StreamPool.
        const auto now = std::chrono::steady_clock::now();
        if (now >= stream_refresh_deadline || model_streams.empty()) {
            model_streams.clear();
            auto streams = pool_.activeStreams();
            model_streams.reserve(streams.size());
            for (const auto& sid : streams) {
                if (pool_.getStreamModelId(sid) == model_cfg_.id) {
                    model_streams.push_back(sid);
                }
            }
            stream_refresh_deadline = now + std::chrono::milliseconds(200);
        }

        // Collect frames from streams assigned to this model
        for (const auto& sid : model_streams) {

            FrameBuffer* buf = pool_.getBuffer(sid);
            if (!buf) continue;

            Frame f;
            if (buf->pop(f)) {
                if (f.meta.model_id.empty()) f.meta.model_id = model_cfg_.id;

                // Enforce a homogeneous batch: once the type is set, reject
                // frames of the opposite type to prevent OOB in the backend.
#ifdef BUILD_ASCEND_BACKEND
                const bool frame_is_ascend = f.is_ascend;
#else
                const bool frame_is_ascend = false;
#endif
                if (!batch.empty()) {
                    const bool type_mismatch =
#ifdef BUILD_ASCEND_BACKEND
                        (frame_is_ascend != batch.is_ascend) ||
#endif
                        (!frame_is_ascend && f.is_gpu != batch.is_gpu);
                    if (type_mismatch) {
                        LOG_WARN("BatchScheduler[{}]: mixed frame types in one batch, "
                                 "dropping frame from stream {}", model_cfg_.id, sid);
                        continue;
                    }
                }

#ifdef BUILD_ASCEND_BACKEND
                if (frame_is_ascend) {
                    batch.ascend_frames.push_back(std::move(f.ascend_buf));
                    batch.is_ascend = true;
                } else
#endif
                if (f.is_gpu) {
                    if (batch.gpu_frames.empty()) {
                        batch.cuda_device_id = f.gpu_buf.cuda_device_id;
                    }
                    batch.gpu_frames.push_back(std::move(f.gpu_buf));
                    batch.is_gpu = true;
                } else {
                    batch.frames.push_back(std::move(f.image));
                }
                batch.metas.push_back(std::move(f.meta));
            }

            if (static_cast<int>(batch.size()) >= max_bs) break;
        }

        bool timeout = (std::chrono::steady_clock::now() >= deadline);

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
                      model_streams.size());
            callback_(std::move(batch));
            batch.frames.clear();
            batch.gpu_frames.clear();
            batch.metas.clear();
            batch.is_gpu = false;
            batch.cuda_device_id = 0;
#ifdef BUILD_ASCEND_BACKEND
            batch.ascend_frames.clear();
            batch.is_ascend = false;
#endif
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::microseconds(delay_us);
        } else if (batch.empty()) {
            // Nothing to do; yield briefly
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            if (timeout) {
                deadline = std::chrono::steady_clock::now() +
                           std::chrono::microseconds(delay_us);
            }
        } else {
            // Non-empty but below preferred size and not timed out yet.
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    LOG_INFO("BatchScheduler[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

} // namespace infer
