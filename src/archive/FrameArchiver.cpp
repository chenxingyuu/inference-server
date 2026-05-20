#include "archive/FrameArchiver.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <sstream>
#include <chrono>

namespace infer {

namespace fs = std::filesystem;
namespace {
bool isSafeStreamId(const std::string& stream_id) {
    if (stream_id.empty()) return false;
    for (const char c : stream_id) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}
}

FrameArchiver::FrameArchiver(FrameArchiveConfig cfg)
    : cfg_(std::move(cfg)) {
    if (!cfg_.enabled) {
        LOG_INFO("FrameArchiver: disabled");
        return;
    }
    LOG_INFO("FrameArchiver: starting local_dir={} interval={} worker_count={}",
             cfg_.local_dir, cfg_.save_interval, cfg_.worker_count);
    workers_.reserve(static_cast<std::size_t>(cfg_.worker_count));
    for (int i = 0; i < cfg_.worker_count; ++i) {
        workers_.emplace_back(&FrameArchiver::workerLoop, this);
    }
}

FrameArchiver::~FrameArchiver() {
    stop_.store(true);
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    LOG_INFO("FrameArchiver: stopped worker_count={}", workers_.size());
}

std::string FrameArchiver::buildObjectKey(const StreamMeta& meta) const {
    if (!isSafeStreamId(meta.stream_id)) {
        throw std::runtime_error("FrameArchiver: invalid stream_id for object key");
    }
    const int64_t ts_ms = static_cast<int64_t>(meta.capture_ts * 1000.0);
    std::ostringstream oss;
    oss << meta.stream_id << "/" << ts_ms << "_" << meta.frame_seq << ".jpg";
    return oss.str();
}

std::string FrameArchiver::buildLocalPath(const StreamMeta& meta) const {
    const std::string key = buildObjectKey(meta);
    fs::path p = fs::path(cfg_.local_dir) / key;
    return p.string();
}

bool FrameArchiver::enqueue(ArchiveTask task) {
    std::unique_lock<std::mutex> lock(mu_);
    if (queue_.size() >= static_cast<std::size_t>(cfg_.queue_capacity)) {
        Metrics::get().incFramesArchiveDropped();
        LOG_WARN("FrameArchiver: queue full (capacity={}) dropping frame", cfg_.queue_capacity);
        return false;
    }
    queue_.push(std::move(task));
    const uint64_t depth = static_cast<uint64_t>(queue_.size());
    lock.unlock();
    cv_.notify_one();
    Metrics::get().setFrameArchiveQueueDepth(depth);
    return true;
}

FrameArchiveResult FrameArchiver::submit(const StreamMeta& meta, const cv::Mat* frame) {
    FrameArchiveResult out;
    if (!isSafeStreamId(meta.stream_id)) {
        Metrics::get().incFramesArchiveDropped();
        out.upload_state = "failed";
        LOG_WARN("FrameArchiver: reject unsafe stream_id={}", meta.stream_id);
        return out;
    }
    if (!cfg_.enabled || frame == nullptr || frame->empty()) {
        out.upload_state = "disabled";
        return out;
    }
    if (cfg_.save_interval > 1 && (meta.frame_seq % static_cast<uint64_t>(cfg_.save_interval) != 0)) {
        out.upload_state = "disabled";
        return out;
    }

    const std::string object_key = buildObjectKey(meta);
    out.local_path = fs::path(cfg_.local_dir).append(object_key).string();
    out.upload_state = "queued";

    ArchiveTask task;
    task.local_path = out.local_path;
    task.object_key = object_key;
    task.frame = frame->clone();
    if (!enqueue(std::move(task))) {
        out.upload_state = "failed";
    }
    return out;
}

void FrameArchiver::workerLoop() {
    while (!stop_.load() || !queue_.empty()) {
        ArchiveTask task;
        std::size_t queue_depth_after_pop = 0;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                return !queue_.empty() || stop_.load();
            });
            if (queue_.empty()) continue;
            task = std::move(queue_.front());
            queue_.pop();
            queue_depth_after_pop = queue_.size();
            Metrics::get().setFrameArchiveQueueDepth(static_cast<uint64_t>(queue_depth_after_pop));
        }

        try {
            const auto write_start = std::chrono::steady_clock::now();
            fs::create_directories(fs::path(task.local_path).parent_path());
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality};
            if (!cv::imwrite(task.local_path, task.frame, params)) {
                Metrics::get().incFramesArchiveDropped();
                LOG_WARN("FrameArchiver: failed to write local frame {}", task.local_path);
                continue;
            }
            const auto write_end = std::chrono::steady_clock::now();
            const auto write_ms = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start).count();
            const auto worker_tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
            Metrics::get().incFramesArchived();
            LOG_DEBUG("FrameArchiver: wrote {} write_ms={} queue_depth_after_pop={} worker_tid={}",
                      task.local_path,
                      write_ms,
                      queue_depth_after_pop,
                      worker_tid);
        } catch (const std::exception& e) {
            Metrics::get().incFramesArchiveDropped();
            LOG_WARN("FrameArchiver: task failed: {}", e.what());
        }
    }
}

} // namespace infer
