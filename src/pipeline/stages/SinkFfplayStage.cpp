#include "pipeline/stages/SinkFfplayStage.h"

#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <limits>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

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
    suppress_output_reopen_.store(true, std::memory_order_relaxed);
    const pid_t pid = ffplay_pid_.load(std::memory_order_relaxed);
    if (pid > 0) {
        ::kill(pid, SIGTERM);
    }
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
    if (!std::isfinite(cfg_.fps) || cfg_.fps <= 0.0 || cfg_.fps > 240.0) {
        LOG_WARN("SinkFfplayStage[{}]: invalid fps={}, skip ffplay launch", id_, cfg_.fps);
        onFfplayFailure();
        return false;
    }

    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) {
        LOG_WARN("SinkFfplayStage[{}]: pipe() failed: {}", id_, std::strerror(errno));
        onFfplayFailure();
        return false;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipe_fd[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_fd[0]);
    posix_spawn_file_actions_addclose(&actions, pipe_fd[1]);

    const std::string video_size = std::to_string(frame.cols) + "x" + std::to_string(frame.rows);
    const std::string framerate = std::to_string(cfg_.fps);
    char* const argv[] = {
        const_cast<char*>("ffplay"),
        const_cast<char*>("-loglevel"), const_cast<char*>("error"),
        const_cast<char*>("-fflags"), const_cast<char*>("nobuffer"),
        const_cast<char*>("-flags"), const_cast<char*>("low_delay"),
        const_cast<char*>("-f"), const_cast<char*>("rawvideo"),
        const_cast<char*>("-pixel_format"), const_cast<char*>("bgr24"),
        const_cast<char*>("-video_size"), const_cast<char*>(video_size.c_str()),
        const_cast<char*>("-framerate"), const_cast<char*>(framerate.c_str()),
        const_cast<char*>("-"),
        nullptr
    };
    pid_t child_pid = -1;
    const int spawn_rc = posix_spawnp(&child_pid, "ffplay", &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipe_fd[0]);
    if (spawn_rc != 0) {
        close(pipe_fd[1]);
        ffplay_pid_.store(-1, std::memory_order_relaxed);
        LOG_WARN("SinkFfplayStage[{}]: failed to launch ffplay (attempt={}): rc={}",
                 id_, reconnect_attempts_.load(std::memory_order_relaxed), spawn_rc);
        onFfplayFailure();
        return false;
    }
    ffplay_pid_.store(child_pid, std::memory_order_relaxed);
    ffplay_pipe_ = fdopen(pipe_fd[1], "w");
    if (ffplay_pipe_ == nullptr) {
        close(pipe_fd[1]);
        if (child_pid > 0) {
            ::kill(child_pid, SIGTERM);
            int status = 0;
            (void)::waitpid(child_pid, &status, 0);
            ffplay_pid_.store(-1, std::memory_order_relaxed);
        }
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
        const int rc = fclose(ffplay_pipe_);
        ffplay_pipe_ = nullptr;
        if (rc != 0) {
            LOG_WARN("SinkFfplayStage[{}]: ffplay exited with status {}", id_, rc);
        }
    }
    const pid_t pid = ffplay_pid_.load(std::memory_order_relaxed);
    if (pid > 0) {
        int status = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        while (std::chrono::steady_clock::now() < deadline) {
            const pid_t rc = ::waitpid(pid, &status, WNOHANG);
            if (rc == pid) {
                ffplay_pid_.store(-1, std::memory_order_relaxed);
                return;
            }
            if (rc < 0) {
                ffplay_pid_.store(-1, std::memory_order_relaxed);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ::kill(pid, SIGKILL);
        (void)::waitpid(pid, &status, 0);
        ffplay_pid_.store(-1, std::memory_order_relaxed);
    }
}

} // namespace infer
