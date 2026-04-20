#pragma once

#include "pipeline/IStage.h"
#include "pipeline/stages/StreamDropPolicy.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core/mat.hpp>

namespace infer {

struct SinkFfplayConfig {
    double fps{15.0};
    int queue_capacity{32};
    int reconnect_initial_ms{1000};
    int reconnect_max_ms{15000};
    float draw_conf_thresh{0.0f};
    int line_thickness{2};
    StreamDropPolicy drop_policy{StreamDropPolicy::DropOldest};
};

class SinkFfplayStage final : public IStage {
public:
    explicit SinkFfplayStage(std::string id, SinkFfplayConfig cfg);
    ~SinkFfplayStage() override;

    std::string id() const override;
    void start() override;
    void stop() override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    struct QueueItem {
        std::shared_ptr<Frame> frame;
        std::optional<InferResult> infer_result;
        std::string stream_id;
    };

    void runWorker();
    void enqueue(QueueItem item);
    bool ensureFfplayOpened(const cv::Mat& frame);
    bool writeFfplayFrame(const cv::Mat& frame);
    void closeFfplay();
    void onFfplayFailure();

    std::string id_;
    SinkFfplayConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<QueueItem> queue_;
    FILE* ffplay_pipe_{nullptr};

    std::atomic<uint64_t> frames_written_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> reconnect_attempts_{0};
    int reconnect_delay_ms_{1000};
    std::chrono::steady_clock::time_point next_reconnect_at_{};
};

} // namespace infer
