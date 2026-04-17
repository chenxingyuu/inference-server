#include "pipeline/stages/DrawAndStreamStage.h"

#include "common/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace infer {

namespace {

cv::Rect clampRect(const BBox& box, int width, int height) {
    const int x1 = std::max(0, std::min(static_cast<int>(std::floor(box.x1)), width - 1));
    const int y1 = std::max(0, std::min(static_cast<int>(std::floor(box.y1)), height - 1));
    const int x2 = std::max(0, std::min(static_cast<int>(std::ceil(box.x2)), width));
    const int y2 = std::max(0, std::min(static_cast<int>(std::ceil(box.y2)), height));
    const int w = std::max(0, x2 - x1);
    const int h = std::max(0, y2 - y1);
    return cv::Rect(x1, y1, w, h);
}

std::string detectionLabel(const Detection& det) {
    std::string label = det.class_name.empty() ? ("cls_" + std::to_string(det.class_id)) : det.class_name;
    label += " " + std::to_string(static_cast<int>(det.confidence * 100.0f)) + "%";
    if (det.track_id.has_value()) {
        label += " #" + std::to_string(*det.track_id);
    }
    return label;
}

std::string buildOutputUrl(const std::string& url, const std::string& protocol) {
    if (protocol != "rtsp") return url;
    if (url.find("rtsp_transport=") != std::string::npos) return url;
    if (url.find('?') == std::string::npos) return url + "?rtsp_transport=tcp";
    return url + "&rtsp_transport=tcp";
}

std::string shellQuote(const std::string& input) {
    std::string out = "'";
    for (const char c : input) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}

} // namespace

DrawAndStreamStage::DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg)
    : DrawAndStreamStage(std::move(id), std::move(cfg), std::make_unique<OpenCvStreamWriter>()) {}

DrawAndStreamStage::DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg, std::unique_ptr<IStreamWriter> writer)
    : id_(std::move(id)), cfg_(std::move(cfg)), writer_(std::move(writer)), reconnect_delay_ms_(cfg_.reconnect_initial_ms) {}

DrawAndStreamStage::~DrawAndStreamStage() { stop(); }

std::string DrawAndStreamStage::id() const { return id_; }

void DrawAndStreamStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    reconnect_delay_ms_ = cfg_.reconnect_initial_ms;
    next_reconnect_at_ = std::chrono::steady_clock::now();
    worker_ = std::thread(&DrawAndStreamStage::runWorker, this);
}

void DrawAndStreamStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    writer_->close();
    std::lock_guard<std::mutex> lock(mu_);
    queue_.clear();
}

void DrawAndStreamStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (input.frame) {
        StreamItem item;
        item.frame = std::make_shared<Frame>(*input.frame);
        item.infer_result = input.infer_result;
        item.stream_id = input.stream_id;
        enqueue(std::move(item));
    }
    emit(input);
}

void DrawAndStreamStage::enqueue(StreamItem item) {
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

void DrawAndStreamStage::runWorker() {
    while (running_.load(std::memory_order_relaxed)) {
        StreamItem item;
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
        drawDetections(output, item.infer_result);

        if (!ensureWriterOpened(output)) {
            continue;
        }

        if (!writer_->write(output)) {
            onWriteFailure();
            continue;
        }
        frames_written_.fetch_add(1, std::memory_order_relaxed);
        consecutive_failures_.store(0, std::memory_order_relaxed);
    }
}

void DrawAndStreamStage::drawDetections(cv::Mat& frame, const std::optional<InferResult>& result) const {
    if (!result.has_value()) return;
    for (const auto& det : result->detections) {
        if (det.confidence < cfg_.draw_conf_thresh) continue;
        cv::Rect rect = clampRect(det.bbox, frame.cols, frame.rows);
        if (rect.width <= 0 || rect.height <= 0) continue;

        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), std::max(1, cfg_.line_thickness));
        const std::string label = detectionLabel(det);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int text_y = std::max(0, rect.y - text_size.height - baseline - 4);
        const cv::Rect bg(rect.x, text_y, std::min(frame.cols - rect.x, text_size.width + 6), text_size.height + baseline + 4);
        cv::rectangle(frame, bg, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(rect.x + 3, text_y + text_size.height + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

bool DrawAndStreamStage::ensureWriterOpened(const cv::Mat& frame) {
    if (writer_->isOpened()) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_reconnect_at_) return false;

    reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);
    if (writer_->open(cfg_.output_url, cfg_.protocol, cfg_.fps, cfg_.bitrate_kbps, frame.cols, frame.rows)) {
        LOG_INFO("DrawAndStreamStage[{}]: stream opened url={} protocol={} size={}x{} fps={}",
                 id_, cfg_.output_url, cfg_.protocol, frame.cols, frame.rows, cfg_.fps);
        reconnect_delay_ms_ = cfg_.reconnect_initial_ms;
        consecutive_failures_.store(0, std::memory_order_relaxed);
        return true;
    }
    LOG_WARN("DrawAndStreamStage[{}]: stream open failed url={} protocol={} (attempt={})",
             id_, cfg_.output_url, cfg_.protocol, reconnect_attempts_.load(std::memory_order_relaxed));
    onWriteFailure();
    return false;
}

void DrawAndStreamStage::onWriteFailure() {
    writer_->close();
    const auto failures = consecutive_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
    next_reconnect_at_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(reconnect_delay_ms_);
    reconnect_delay_ms_ = std::min(reconnect_delay_ms_ * 2, std::max(cfg_.reconnect_max_ms, cfg_.reconnect_initial_ms));
    LOG_WARN("DrawAndStreamStage[{}]: stream write failed, failures={}, next_reconnect_in_ms={}",
             id_, failures, reconnect_delay_ms_);
}

bool DrawAndStreamStage::OpenCvStreamWriter::open(
    const std::string& url,
    const std::string& protocol,
    double fps,
    int,
    int width,
    int height) {
    close();
    if (url.empty()) return false;
    const std::string output_url = buildOutputUrl(url, protocol);
    std::string format = "rtsp";
    if (protocol == "rtmp") format = "flv";
    const std::string cmd =
        "ffmpeg -loglevel error -nostats "
        "-f rawvideo -pix_fmt bgr24 "
        "-s " + std::to_string(width) + "x" + std::to_string(height) + " "
        "-r " + std::to_string(fps) + " -i - "
        "-an -c:v libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency "
        "-f " + format + " " + shellQuote(output_url);

    pipe_ = popen(cmd.c_str(), "w");
    return pipe_ != nullptr;
}

bool DrawAndStreamStage::OpenCvStreamWriter::write(const cv::Mat& frame) {
    if (pipe_ == nullptr) return false;
    if (frame.empty()) return false;
    if (!frame.isContinuous()) {
        for (int row = 0; row < frame.rows; ++row) {
            const auto* row_ptr = frame.ptr<uint8_t>(row);
            const std::size_t wrote = std::fwrite(row_ptr, 1, static_cast<std::size_t>(frame.cols * frame.elemSize()), pipe_);
            if (wrote != static_cast<std::size_t>(frame.cols * frame.elemSize())) return false;
        }
        return std::fflush(pipe_) == 0;
    }
    const std::size_t bytes = frame.total() * frame.elemSize();
    const std::size_t wrote = std::fwrite(frame.data, 1, bytes, pipe_);
    return (wrote == bytes && std::fflush(pipe_) == 0);
}

void DrawAndStreamStage::OpenCvStreamWriter::close() {
    if (pipe_ != nullptr) {
        pclose(pipe_);
        pipe_ = nullptr;
    }
}

bool DrawAndStreamStage::OpenCvStreamWriter::isOpened() const { return pipe_ != nullptr; }

} // namespace infer
