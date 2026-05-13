#include "pipeline/stages/DrawAndStreamStage.h"

#ifdef BUILD_ASCEND_BACKEND
#include "pipeline/stages/AscendVencFfmpegMuxWriter.h"
#endif

#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace infer {

namespace {

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

uint64_t nowSteadyNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

double nsToMs(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

double steadyMsBetween(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(b - a).count();
}

} // namespace

namespace {

std::unique_ptr<IStreamWriter> makeSinkStreamWriter(const DrawAndStreamConfig& cfg, int ctx_ascend_device_id) {
#ifdef BUILD_ASCEND_BACKEND
    if (cfg.video_encoder == StreamVideoEncoder::AscendVenc) {
        const int dev = cfg.ascend_device_id >= 0 ? cfg.ascend_device_id : ctx_ascend_device_id;
        return std::make_unique<AscendVencFfmpegMuxWriter>(dev);
    }
#else
    if (cfg.video_encoder == StreamVideoEncoder::AscendVenc) {
        throw std::runtime_error("sink.stream encoder ascend_venc requires BUILD_ASCEND_BACKEND=ON");
    }
#endif
    return std::make_unique<DrawAndStreamStage::OpenCvStreamWriter>();
}

} // namespace

DrawAndStreamStage::DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg)
    : DrawAndStreamStage(std::move(id), std::move(cfg), std::make_unique<DrawAndStreamStage::OpenCvStreamWriter>()) {}

DrawAndStreamStage::DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg, int ctx_ascend_device_id)
    : DrawAndStreamStage(std::move(id), std::move(cfg), makeSinkStreamWriter(cfg, ctx_ascend_device_id)) {}

DrawAndStreamStage::DrawAndStreamStage(std::string id, DrawAndStreamConfig cfg, std::unique_ptr<IStreamWriter> writer)
    : id_(std::move(id))
    , cfg_(std::move(cfg))
    , writer_(std::move(writer))
    , reconnect_delay_ms_(cfg_.reconnect_initial_ms) {}

DrawAndStreamStage::~DrawAndStreamStage() { stop(); }

std::string DrawAndStreamStage::id() const { return id_; }

void DrawAndStreamStage::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    suppress_output_reopen_.store(false, std::memory_order_relaxed);
    reconnect_delay_ms_ = cfg_.reconnect_initial_ms;
    next_reconnect_at_ = std::chrono::steady_clock::now();
    worker_ = std::thread(&DrawAndStreamStage::runWorker, this);
}

void DrawAndStreamStage::onGraphExecutorDraining() noexcept {
    suppress_output_reopen_.store(true, std::memory_order_relaxed);
    cv_.notify_all();
}

void DrawAndStreamStage::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join(); // runWorker() closes writer_ on exit — safe to clear queue after join
    }
    std::lock_guard<std::mutex> lock(mu_);
    queue_.clear();
}

void DrawAndStreamStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (input.frame) {
        StreamItem item;
        item.frame = input.frame; // shared ownership; worker clones before drawing
        item.infer_result = input.infer_result;
        item.stream_id = input.stream_id;
        enqueue(std::move(item));
    }
    emit(input);
}

void DrawAndStreamStage::enqueue(StreamItem item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_.load(std::memory_order_relaxed)) return; // stage is stopped, discard
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

        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat output = item.frame->image.clone();
        const auto t1 = std::chrono::steady_clock::now();
        overlay::drawDetections(output, item.infer_result, cfg_.draw_conf_thresh, cfg_.line_thickness);
        if (cfg_.draw_timestamp) {
            overlay::drawTimestamp(
                output,
                item.stream_id,
                cfg_.draw_timestamp_with_stream_id,
                cfg_.timestamp_x,
                cfg_.timestamp_y,
                cfg_.timestamp_font_scale,
                cfg_.timestamp_thickness);
        }
        const auto t2 = std::chrono::steady_clock::now();

        const char* skip_reason = nullptr;
        int         reconnect_wait_ms = 0;
        if (!ensureWriterOpened(output, &skip_reason, &reconnect_wait_ms)) {
            const auto t_fail = std::chrono::steady_clock::now();
            const bool in_backoff =
                skip_reason && std::strcmp(skip_reason, "reconnect_backoff") == 0;
            if (in_backoff) {
                using namespace std::chrono;
                if (t_fail - last_backoff_timing_debug_log_ < seconds(3)) {
                    continue;
                }
                last_backoff_timing_debug_log_ = t_fail;
            }
            const std::size_t dets = item.infer_result.has_value() ? item.infer_result->detections.size() : 0;
            const uint64_t frame_seq = item.infer_result.has_value() ? item.infer_result->frame_seq : 0;
            LOG_DEBUG(
                "DrawAndStreamStage[{}]: timing stream={} frame_seq={} size={}x{} dets={} "
                "clone_ms={:.2f} draw_ms={:.2f} ensure_open_ms={:.2f} skip={} reconnect_wait_ms={} total_ms={:.2f}",
                id_,
                item.stream_id,
                frame_seq,
                output.cols,
                output.rows,
                dets,
                steadyMsBetween(t0, t1),
                steadyMsBetween(t1, t2),
                steadyMsBetween(t2, t_fail),
                skip_reason ? skip_reason : "?",
                reconnect_wait_ms,
                steadyMsBetween(t0, t_fail));
            continue;
        }
        const auto t3 = std::chrono::steady_clock::now();

        if (!writer_->write(output)) {
            const auto t_fail = std::chrono::steady_clock::now();
            const std::size_t dets = item.infer_result.has_value() ? item.infer_result->detections.size() : 0;
            const uint64_t frame_seq = item.infer_result.has_value() ? item.infer_result->frame_seq : 0;
            LOG_DEBUG(
                "DrawAndStreamStage[{}]: timing stream={} frame_seq={} size={}x{} dets={} "
                "clone_ms={:.2f} draw_ms={:.2f} ensure_open_ms={:.2f} write_ms={:.2f} (write failed) total_ms={:.2f}",
                id_,
                item.stream_id,
                frame_seq,
                output.cols,
                output.rows,
                dets,
                steadyMsBetween(t0, t1),
                steadyMsBetween(t1, t2),
                steadyMsBetween(t2, t3),
                steadyMsBetween(t3, t_fail),
                steadyMsBetween(t0, t_fail));
            onStreamFailure("write");
            continue;
        }
        const auto t4 = std::chrono::steady_clock::now();
        frames_written_.fetch_add(1, std::memory_order_relaxed);
        if (item.infer_result.has_value() && item.infer_result->frame_mono_ns != 0) {
            const uint64_t now_ns = nowSteadyNs();
            if (now_ns >= item.infer_result->frame_mono_ns) {
                Metrics::get().recordSinkStreamLatency(
                    item.stream_id,
                    nsToMs(now_ns - item.infer_result->frame_mono_ns));
            }
        }
        const double latency_ms = item.infer_result.has_value() ? item.infer_result->latency_ms : -1.0;
        const std::size_t dets = item.infer_result.has_value() ? item.infer_result->detections.size() : 0;
        const uint64_t frame_seq = item.infer_result.has_value() ? item.infer_result->frame_seq : 0;
        LOG_DEBUG(
            "DrawAndStreamStage[{}]: timing stream={} frame_seq={} size={}x{} dets={} "
            "clone_ms={:.2f} draw_ms={:.2f} ensure_open_ms={:.2f} write_ms={:.2f} total_ms={:.2f} infer_latency_ms={:.1f}",
            id_,
            item.stream_id,
            frame_seq,
            output.cols,
            output.rows,
            dets,
            steadyMsBetween(t0, t1),
            steadyMsBetween(t1, t2),
            steadyMsBetween(t2, t3),
            steadyMsBetween(t3, t4),
            steadyMsBetween(t0, t4),
            latency_ms);
        consecutive_failures_.store(0, std::memory_order_relaxed);
    }
    // writer_ is exclusively owned by this thread: close it here so stop() never
    // races with an in-progress write.
    writer_->close();
}

bool DrawAndStreamStage::ensureWriterOpened(const cv::Mat& frame, const char** out_skip_reason, int* out_reconnect_wait_ms) {
    if (writer_->isOpened()) return true;
    if (suppress_output_reopen_.load(std::memory_order_relaxed)) {
        if (out_skip_reason) *out_skip_reason = "suppressed_graph_draining";
        if (out_reconnect_wait_ms) *out_reconnect_wait_ms = 0;
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next_reconnect_at_) {
        if (out_skip_reason) *out_skip_reason = "reconnect_backoff";
        if (out_reconnect_wait_ms) {
            *out_reconnect_wait_ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(next_reconnect_at_ - now).count());
        }
        return false;
    }

    reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);
    const int gop = (cfg_.gop > 0) ? cfg_.gop : std::max(1, static_cast<int>(std::lround(cfg_.fps)));
    if (writer_->open(cfg_.output_url, cfg_.protocol, cfg_.fps, gop, cfg_.bitrate_kbps, frame.cols, frame.rows)) {
        LOG_INFO("DrawAndStreamStage[{}]: stream opened url={} protocol={} size={}x{} fps={}",
                 id_, cfg_.output_url, cfg_.protocol, frame.cols, frame.rows, cfg_.fps);
        reconnect_delay_ms_ = cfg_.reconnect_initial_ms;
        consecutive_failures_.store(0, std::memory_order_relaxed);
        last_backoff_timing_debug_log_ = {};
        return true;
    }
    LOG_WARN("DrawAndStreamStage[{}]: stream open failed url={} protocol={} (attempt={})",
             id_, cfg_.output_url, cfg_.protocol, reconnect_attempts_.load(std::memory_order_relaxed));
    if (out_skip_reason) *out_skip_reason = "open_failed";
    if (out_reconnect_wait_ms) *out_reconnect_wait_ms = 0;
    onStreamFailure("open");
    return false;
}

void DrawAndStreamStage::onStreamFailure(const char* phase) {
    writer_->close();
    if (suppress_output_reopen_.load(std::memory_order_relaxed)) return;
    const auto failures = consecutive_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
    const int  backoff_ms = reconnect_delay_ms_;
    next_reconnect_at_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(backoff_ms);
    reconnect_delay_ms_ = std::min(reconnect_delay_ms_ * 2, std::max(cfg_.reconnect_max_ms, cfg_.reconnect_initial_ms));
    LOG_WARN("DrawAndStreamStage[{}]: stream {} failed, failures={}, scheduled_backoff_ms={}, next_backoff_ms={}",
             id_,
             phase,
             failures,
             backoff_ms,
             reconnect_delay_ms_);
}

bool DrawAndStreamStage::OpenCvStreamWriter::open(
    const std::string& url,
    const std::string& protocol,
    double fps,
    int gop,
    int bitrate_kbps,
    int width,
    int height) {
    close();
    if (url.empty()) return false;
    // Defense-in-depth: reject control characters that would break the shell command.
    for (unsigned char c : url) {
        if (c < 0x20 || c == 0x7f) return false;
    }
    const std::string output_url = buildOutputUrl(url, protocol);
    std::string format = "rtsp";
    if (protocol == "rtmp") format = "flv";
    const std::string cmd =
        "ffmpeg -loglevel error -nostats "
        "-f rawvideo -pix_fmt bgr24 "
        "-s " + std::to_string(width) + "x" + std::to_string(height) + " "
        "-r " + std::to_string(fps) + " -i - "
        "-an -c:v libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency "
        "-g " + std::to_string(gop) + " -keyint_min " + std::to_string(gop) + " -sc_threshold 0 "
        "-b:v " + std::to_string(bitrate_kbps) + "k "
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
        const int rc = pclose(pipe_);
        pipe_ = nullptr;
        if (rc != 0) {
            // Log via stderr since we have no id_ in this inner class.
            // Callers (DrawAndStreamStage) will log context around failures.
            std::fprintf(stderr, "[DrawAndStreamStage] ffmpeg exited with status %d\n", rc);
        }
    }
}

bool DrawAndStreamStage::OpenCvStreamWriter::isOpened() const { return pipe_ != nullptr; }

} // namespace infer
