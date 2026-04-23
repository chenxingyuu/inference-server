#pragma once

#include "common/Config.h"
#include "decoder/IYOLODecoder.h"
#include "infer/IInferBackend.h"
#include "pipeline/IStage.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace infer {

class InferEngineStage final : public IStage {
public:
    InferEngineStage(std::string id,
                     ModelConfig model_cfg,
                     std::unique_ptr<IInferBackend> backend,
                     std::unique_ptr<IYOLODecoder> decoder);

    std::string id() const override;

    void start() override;
    void stop() override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    // Under pending_mutex_: pop up to flush_count frames and reset deadline.
    // Returns the extracted frames; caller must NOT hold pending_mutex_ afterwards
    // when calling inferAndEmit().
    std::vector<EventEnvelope> extractBatch(int flush_count);

    // Run TRT inference + decode + emit on an already-extracted batch.
    // Must NOT be called with pending_mutex_ held (emit can block on downstream queues).
    // Serialized internally by infer_mutex_ to prevent concurrent GPU calls.
    void inferAndEmit(std::vector<EventEnvelope> events, const EmitFn& emit);

    // Background thread: fires every max_queue_delay_us/2 to flush stale frames.
    void flushLoop();

    std::string id_;
    ModelConfig model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder> decoder_;

    std::mutex pending_mutex_;              // guards pending_events_, batch_deadline_, last_emit_
    std::mutex infer_mutex_;               // serializes GPU inference between process() and flushLoop()
    std::vector<EventEnvelope> pending_events_;
    std::chrono::steady_clock::time_point batch_deadline_;
    EmitFn last_emit_;

    std::thread flush_thread_;
    std::atomic<bool> flush_stop_{false};
    std::condition_variable flush_cv_;

    int max_pending_{0};
    bool fallback_to_single_infer_{false};
    bool loaded_{false};
};

} // namespace infer
