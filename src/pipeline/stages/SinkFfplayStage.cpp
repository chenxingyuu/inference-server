#include "pipeline/stages/SinkFfplayStage.h"

#include "common/Logger.h"
#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace infer {

SinkFfplayStage::SinkFfplayStage(std::string id, SinkFfplayConfig cfg)
    : id_(std::move(id))
    , cfg_(std::move(cfg)) {}

SinkFfplayStage::~SinkFfplayStage() { stop(); }

std::string SinkFfplayStage::id() const { return id_; }

void SinkFfplayStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    worker_ = std::thread(&SinkFfplayStage::runWorker, this);
}

void SinkFfplayStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    closeFfplay();
    std::lock_guard<std::mutex> lock(mu_);
    queue_.clear();
}

void SinkFfplayStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (input.frame) {
        QueueItem item;
        item.frame = std::make_shared<Frame>(*input.frame);
        item.infer_result = input.infer_result;
        item.stream_id = input.stream_id;
        enqueue(std::move(item));
    }
    emit(input);
}

void SinkFfplayStage::enqueue(QueueItem item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.size() >= static_cast<std::size_t>(std::max(1, cfg_.queue_capacity))) {
        if (cfg_.drop_policy == StreamDropPolicy::DropOldest) {
            queue_.pop_front();
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
        } else {
            frames_dropped_.fetch_add(1, std::memory_order_relaxed);
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
        }
        if (!item.frame || item.frame->image.empty()) {
            continue;
        }

        cv::Mat output = item.frame->image.clone();
        overlay::drawDetections(output, item.infer_result, cfg_.draw_conf_thresh, cfg_.line_thickness);

        if (!ensureFfplayOpened(output) || !writeFfplayFrame(output)) {
            closeFfplay();
            continue;
        }
        frames_written_.fetch_add(1, std::memory_order_relaxed);
    }
}

bool SinkFfplayStage::ensureFfplayOpened(const cv::Mat& frame) {
    if (ffplay_pipe_ != nullptr) return true;
    const std::string cmd =
        "ffplay -loglevel error -fflags nobuffer -flags low_delay "
        "-f rawvideo -pixel_format bgr24 "
        "-video_size " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) + " "
        "-framerate " + std::to_string(cfg_.fps) + " -";
    ffplay_pipe_ = popen(cmd.c_str(), "w");
    if (ffplay_pipe_ == nullptr) {
        LOG_WARN("SinkFfplayStage[{}]: failed to launch ffplay", id_);
        return false;
    }
    LOG_INFO("SinkFfplayStage[{}]: ffplay preview started", id_);
    return true;
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
        pclose(ffplay_pipe_);
        ffplay_pipe_ = nullptr;
    }
}

} // namespace infer
