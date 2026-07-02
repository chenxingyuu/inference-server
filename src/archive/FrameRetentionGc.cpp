#include "archive/FrameRetentionGc.h"
#include "archive/FrameLayout.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include <filesystem>
#include <chrono>

namespace infer {

namespace fs = std::filesystem;

FrameRetentionGc::FrameRetentionGc(FrameArchiveConfig cfg)
    : cfg_(std::move(cfg)) {}

FrameRetentionGc::~FrameRetentionGc() {
    stop();
}

void FrameRetentionGc::start() {
    if (!cfg_.retention.enabled) {
        LOG_INFO("FrameRetentionGc: disabled");
        return;
    }
    std::lock_guard lock(mu_);
    if (started_) return;
    stop_.store(false);
    worker_ = std::thread(&FrameRetentionGc::loop, this);
    started_ = true;
    LOG_INFO("FrameRetentionGc: started local_dir={} max_age_minutes={} scan_interval_seconds={}",
             cfg_.local_dir,
             cfg_.retention.max_age_minutes,
             cfg_.retention.scan_interval_seconds);
}

void FrameRetentionGc::stop() {
    if (!started_) return;
    stop_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    started_ = false;
    LOG_INFO("FrameRetentionGc: stopped");
}

void FrameRetentionGc::loop() {
    while (!stop_.load()) {
        {
            std::unique_lock lock(mu_);
            cv_.wait_for(lock,
                         std::chrono::seconds(cfg_.retention.scan_interval_seconds),
                         [this] { return stop_.load(); });
            if (stop_.load()) break;
        }
        const std::time_t now = std::time(nullptr);
        runOnce(now);
    }
}

std::uintmax_t FrameRetentionGc::runOnce(std::time_t now) {
    if (!cfg_.retention.enabled) return 0;

    const fs::path root(cfg_.local_dir);
    if (!fs::exists(root)) return 0;

    const std::time_t cutoff = now - cfg_.retention.max_age_minutes * 60;
    std::uintmax_t removed = 0;

    auto removeTree = [&](const fs::path& p) {
        try {
            if (!fs::exists(p)) return;
            const auto n = fs::remove_all(p);
            removed += n;
            Metrics::get().incFramesArchiveDeleted(n);
            LOG_DEBUG("FrameRetentionGc: removed {} entries under {}", n, p.string());
        } catch (const std::exception& e) {
            LOG_WARN("FrameRetentionGc: failed to remove {}: {}", p.string(), e.what());
        }
    };

    auto removeIfEmpty = [&](const fs::path& p) {
        try {
            if (fs::exists(p) && fs::is_directory(p) && fs::is_empty(p)) {
                fs::remove(p);
            }
        } catch (const std::exception& e) {
            LOG_WARN("FrameRetentionGc: failed to prune empty dir {}: {}", p.string(), e.what());
        }
    };

    try {
        for (const auto& day_entry : fs::directory_iterator(root)) {
            if (!day_entry.is_directory()) continue;
            const auto day_name = day_entry.path().filename().string();
            const auto day_start = frame_layout::parseDayDir(day_name);
            if (!day_start.has_value()) continue;

            const std::time_t day_end = *day_start + 86400;
            if (day_end <= cutoff) {
                removeTree(day_entry.path());
                continue;
            }

            for (const auto& hour_entry : fs::directory_iterator(day_entry.path())) {
                if (!hour_entry.is_directory()) continue;
                const auto hour_name = hour_entry.path().filename().string();
                const auto hour_opt = frame_layout::parseHourDir(hour_name);
                if (!hour_opt.has_value()) continue;

                const std::time_t hour_start = *day_start + static_cast<std::time_t>(*hour_opt) * 3600;
                const std::time_t hour_end = hour_start + 3600;
                if (hour_end <= cutoff) {
                    removeTree(hour_entry.path());
                    continue;
                }

                for (const auto& minute_entry : fs::directory_iterator(hour_entry.path())) {
                    if (!minute_entry.is_directory()) continue;
                    const auto minute_name = minute_entry.path().filename().string();
                    const auto minute_opt = frame_layout::parseMinuteDir(minute_name);
                    if (!minute_opt.has_value()) continue;

                    const std::time_t minute_start =
                        hour_start + static_cast<std::time_t>(*minute_opt) * 60;
                    const std::time_t minute_end = minute_start + 60;
                    if (minute_end <= cutoff) {
                        removeTree(minute_entry.path());
                    }
                }

                removeIfEmpty(hour_entry.path());
            }

            removeIfEmpty(day_entry.path());
        }
    } catch (const std::exception& e) {
        LOG_WARN("FrameRetentionGc: scan failed: {}", e.what());
    }

    if (removed > 0) {
        LOG_INFO("FrameRetentionGc: removed {} entries (cutoff={})", removed, cutoff);
    }
    return removed;
}

} // namespace infer
