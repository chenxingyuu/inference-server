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
    // Runs inference on pending_events_[0..flush_count). Called with pending_mutex_ held.
    void doFlush(int flush_count, const EmitFn& emit);
    // Background thread: flushes pending batch when deadline expires between frame arrivals.
    void flushLoop();

    std::string id_;
    ModelConfig model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder> decoder_;

    std::mutex pending_mutex_;
    std::vector<EventEnvelope> pending_events_;
    std::chrono::steady_clock::time_point batch_deadline_;
    EmitFn last_emit_;          // updated each process() call; used by flush thread

    std::thread flush_thread_;
    std::atomic<bool> flush_stop_{false};
    std::condition_variable flush_cv_;

    int max_pending_{0};
    bool fallback_to_single_infer_{false};
    bool loaded_{false};
};

} // namespace infer
