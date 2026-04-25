#include "pipeline/stages/SinkFfplayStage.h"

#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace infer {

SinkFfplayStage::SinkFfplayStage(std::string id, SinkFfplayConfig cfg)
    : id_(std::move(id))
    , cfg_(std::move(cfg))
    , reconnect_delay_ms_(cfg_.reconnect_initial_ms) {}

SinkFfplayStage::~SinkFfplayStage() { stop(); }

std::string SinkFfplayStage::id() const { return id_; }

void SinkFfplayStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    suppress_output_reopen_.store(false, std::memory_order_relaxed);
    worker_ = std::thread(&SinkFfplayStage::runWorker, this);
}

void SinkFfplayStage::onGraphExecutorDraining() noexcept {
    suppress_output_reopen_.store(true, std::memory_order_relaxed);
    cv_.notify_all();
}

void SinkFfplayStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join(); // runWorker() closes ffplay_pipe_ on exit — safe to clear queue after join
    }
    std::lock_guard<std::mutex> lock(mu_);
    queue_.clear();
}

void SinkFfplayStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (input.frame) {
        QueueItem item;
        item.frame = input.frame; // shared ownership; worker clones before drawing
        item.infer_result = input.infer_result;
        item.stream_id = input.stream_id;
        enqueue(std::move(item));
    }
    emit(input);
}

void SinkFfplayStage::enqueue(QueueItem item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_.load(std::memory_order_relaxed)) return; // stage is stopped, discard
    if (queue_.size() >= static_cast<std::size_t>(std::max(1, cfg_.queue_capacity))) {
        if (cfg_.drop_policy == StreamDropPolicy::DropOldest) {
            queue_.pop_front();
            Metrics::get().incSinkFfplayFramesDropped(id_);
        } else {
            Metrics::get().incSinkFfplayFramesDropped(id_);
            return;
        }
    }
    queue_.push_back(std::move(item));
    cv_.notify_one();
}

void SinkFfplayStage::runWorker() {
    while (running_.load(std::memory_order_relaxed)) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] {
                return !running_.load(std::memory_order_relaxed) || !queue_.empty();
            });
            if (!running_.load(std::memory_order_relaxed) && queue_.empty()) {
                break;
            }
            item = std::move(queue_.front());
            queue_.pop_front();
            Metrics::get().setSinkFfplayQueueDepth(id_, queue_.size());
        }
        if (!item.frame || item.frame->image.empty()) {
            continue;
        }

        cv::Mat output = item.frame->image.clone();
        overlay::drawDetections(output, item.infer_result, cfg_.draw_conf_thresh, cfg_.line_thickness);

        if (!ensureFfplayOpened(output)) {
            continue;
        }
        if (!writeFfplayFrame(output)) {
            closeFfplay();
            onFfplayFailure();
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_write_at_.has_value()) {
            const double ms = std::chrono::duration<double, std::milli>(now - *last_write_at_).count();
            Metrics::get().recordSinkFfplayJitter(id_, ms);
        }
        last_write_at_ = now;
        Metrics::get().incSinkFfplayFramesWritten(id_);
    }
    // ffplay_pipe_ is exclusively owned by this thread: close here so stop() never
    // races with an in-progress fwrite.
    closeFfplay();
    last_write_at_.reset();
}

bool SinkFfplayStage::ensureFfplayOpened(const cv::Mat& frame) {
    if (ffplay_pipe_ != nullptr) return true;
    if (suppress_output_reopen_.load(std::memory_order_relaxed)) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_reconnect_at_) return false;

    reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);
    // Clamp fps to a safe integer range before embedding in the shell command.
    const int safe_fps = std::max(1, std::min(static_cast<int>(cfg_.fps), 240));
    const std::string cmd =
        "ffplay -loglevel error -fflags nobuffer -flags low_delay "
        "-f rawvideo -pixel_format bgr24 "
        "-video_size " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) + " "
        "-framerate " + std::to_string(safe_fps) + " -";
    ffplay_pipe_ = popen(cmd.c_str(), "w");
    if (ffplay_pipe_ == nullptr) {
        LOG_WARN("SinkFfplayStage[{}]: failed to launch ffplay (attempt={})",
                 id_, reconnect_attempts_.load(std::memory_order_relaxed));
        onFfplayFailure();
        return false;
    }
    LOG_INFO("SinkFfplayStage[{}]: ffplay preview started", id_);
    reconnect_delay_ms_ = cfg_.reconnect_initial_ms;
    return true;
}

void SinkFfplayStage::onFfplayFailure() {
    if (suppress_output_reopen_.load(std::memory_order_relaxed)) return;
    next_reconnect_at_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(reconnect_delay_ms_);
    reconnect_delay_ms_ = std::min(reconnect_delay_ms_ * 2, cfg_.reconnect_max_ms);
    LOG_WARN("SinkFfplayStage[{}]: next ffplay reconnect in {}ms", id_, reconnect_delay_ms_);
}

bool SinkFfplayStage::writeFfplayFrame(const cv::Mat& frame) {
    if (ffplay_pipe_ == nullptr || frame.empty()) return false;
    if (!frame.isContinuous()) {
        for (int row = 0; row < frame.rows; ++row) {
            const auto* row_ptr = frame.ptr<uint8_t>(row);
            const std::size_t wrote = std::fwrite(row_ptr, 1, static_cast<std::size_t>(frame.cols * frame.elemSize()), ffplay_pipe_);
            if (wrote != static_cast<std::size_t>(frame.cols * frame.elemSize())) return false;
        }
        return std::fflush(ffplay_pipe_) == 0;
    }
    const std::size_t bytes = frame.total() * frame.elemSize();
    const std::size_t wrote = std::fwrite(frame.data, 1, bytes, ffplay_pipe_);
    return (wrote == bytes && std::fflush(ffplay_pipe_) == 0);
}

void SinkFfplayStage::closeFfplay() {
    if (ffplay_pipe_ != nullptr) {
        const int rc = pclose(ffplay_pipe_);
        ffplay_pipe_ = nullptr;
        if (rc != 0) {
            LOG_WARN("SinkFfplayStage[{}]: ffplay exited with status {}", id_, rc);
        }
    }
}

} // namespace infer
