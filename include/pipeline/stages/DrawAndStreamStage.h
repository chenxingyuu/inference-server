#pragma once

#include "pipeline/IStage.h"
#include "pipeline/stages/StreamDropPolicy.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <cstdio>

#include <opencv2/core/mat.hpp>

namespace infer {

struct DrawAndStreamConfig {
    std::string output_url;
    std::string protocol{"rtsp"};
    double fps{15.0};
    int gop{0}; // 0 => auto (rounded fps)
    int bitrate_kbps{2500};
    int queue_capacity{32};
    int reconnect_initial_ms{1000};
    int reconnect_max_ms{15000};
    float draw_conf_thresh{0.0f};
    int line_thickness{2};
    StreamDropPolicy drop_policy{StreamDropPolicy::DropOldest};
};

class IStreamWriter {
public:
    virtual ~IStreamWriter() = default;
    virtual bool open(const std::string& url, const std::string& protocol, double fps, int gop, int bitrate_kbps, int width, int height) = 0;
    virtual bool write(const cv::Mat& frame) = 0;
    virtual void close() = 0;
    virtual bool isOpened() const = 0;
};

class DrawAndStreamStage final : public IStage {
public:
    DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg);
    DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg, std::unique_ptr<IStreamWriter> writer);
    ~DrawAndStreamStage() override;

    std::string id() const override;
    void start() override;
    void stop() override;
    void onGraphExecutorDraining() noexcept override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    struct StreamItem {
        std::shared_ptr<Frame> frame;
        std::optional<InferResult> infer_result;
        std::string stream_id;
    };

    class OpenCvStreamWriter final : public IStreamWriter {
    public:
        bool open(const std::string& url, const std::string& protocol, double fps, int gop, int bitrate_kbps, int width, int height) override;
        bool write(const cv::Mat& frame) override;
        void close() override;
        bool isOpened() const override;

    private:
        FILE* pipe_{nullptr};
    };

    void runWorker();
    void enqueue(StreamItem item);
    bool ensureWriterOpened(const cv::Mat& frame);
    void onWriteFailure();

    std::string id_;
    DrawAndStreamConfig cfg_;
    std::unique_ptr<IStreamWriter> writer_;
    std::atomic<bool> suppress_output_reopen_{false};
    std::atomic<bool> running_{false};
    std::thread worker_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<StreamItem> queue_;
    int reconnect_delay_ms_{1000};
    std::chrono::steady_clock::time_point next_reconnect_at_{};

    std::atomic<uint64_t> frames_written_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> reconnect_attempts_{0};
    std::atomic<uint64_t> consecutive_failures_{0};
};

} // namespace infer
