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

enum class StreamVideoEncoder {
    FfmpegLibx264,
    AscendVenc,
};

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
    StreamVideoEncoder video_encoder{StreamVideoEncoder::FfmpegLibx264};
    /// Used when video_encoder == AscendVenc; if < 0, StageFactory uses Context::ingest_ascend_device_id.
    int ascend_device_id{-1};
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
    DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg, int ctx_ascend_device_id);
    ~DrawAndStreamStage() override;

    std::string id() const override;
    void start() override;
    void stop() override;
    void onGraphExecutorDraining() noexcept override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

    class OpenCvStreamWriter final : public IStreamWriter {
    public:
        bool open(const std::string& url, const std::string& protocol, double fps, int gop, int bitrate_kbps, int width, int height) override;
        bool write(const cv::Mat& frame) override;
        void close() override;
        bool isOpened() const override;

    private:
        FILE* pipe_{nullptr};
    };

private:
    struct StreamItem {
        std::shared_ptr<Frame> frame;
        std::optional<InferResult> infer_result;
        std::string stream_id;
    };

    void runWorker();
    void enqueue(StreamItem item);
    /// @param out_skip_reason if non-null and return false, set to a short static reason string.
    /// @param out_reconnect_wait_ms if non-null and in backoff, set to ms until the next open attempt.
    bool ensureWriterOpened(const cv::Mat& frame, const char** out_skip_reason = nullptr, int* out_reconnect_wait_ms = nullptr);
    void onStreamFailure(const char* phase); // phase: "open" or "write"

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
    /// Throttle per-frame DEBUG while waiting for reconnect (avoid log floods at video rate).
    std::chrono::steady_clock::time_point last_backoff_timing_debug_log_{};

    std::atomic<uint64_t> frames_written_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> reconnect_attempts_{0};
    std::atomic<uint64_t> consecutive_failures_{0};
};

} // namespace infer
