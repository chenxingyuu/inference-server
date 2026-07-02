#include "archive/FrameRetentionGc.h"
#include "archive/FrameLayout.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace infer {
namespace {

std::string makeTempDir(const std::string& suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path p = std::filesystem::temp_directory_path() /
                              ("frame_retention_gc_" + suffix + "_" + std::to_string(now));
    std::filesystem::create_directories(p);
    return p.string();
}

void touchFile(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "x";
}

FrameArchiveConfig makeCfg(const std::string& local_dir, bool retention_enabled = true) {
    FrameArchiveConfig cfg;
    cfg.local_dir = local_dir;
    cfg.retention.enabled = retention_enabled;
    cfg.retention.max_age_minutes = 60;
    cfg.retention.scan_interval_seconds = 60;
    return cfg;
}

// 2026-07-02 10:30 UTC
constexpr std::time_t kNow = 1782988200;

void createMinuteBucket(const std::filesystem::path& root,
                        int year, int month, int day,
                        int hour, int minute,
                        const std::string& stream_id) {
    char day_buf[9];
    std::snprintf(day_buf, sizeof(day_buf), "%04d%02d%02d", year, month, day);
    char hour_buf[3];
    std::snprintf(hour_buf, sizeof(hour_buf), "%02d", hour);
    char minute_buf[3];
    std::snprintf(minute_buf, sizeof(minute_buf), "%02d", minute);
    const auto file = root / day_buf / hour_buf / minute_buf / stream_id / "1_1.jpg";
    touchFile(file);
}

bool bucketExists(const std::filesystem::path& root,
                  int year, int month, int day,
                  int hour, int minute) {
    char day_buf[9];
    std::snprintf(day_buf, sizeof(day_buf), "%04d%02d%02d", year, month, day);
    char hour_buf[3];
    std::snprintf(hour_buf, sizeof(hour_buf), "%02d", hour);
    char minute_buf[3];
    std::snprintf(minute_buf, sizeof(minute_buf), "%02d", minute);
    return std::filesystem::exists(root / day_buf / hour_buf / minute_buf);
}

} // namespace

TEST(FrameRetentionGcTest, DeletesExpiredMinuteBucketsOnly) {
    const auto root = makeTempDir("ttl");
    auto cfg = makeCfg(root);
    FrameRetentionGc gc(cfg);

    // Expired: 09:29 bucket (ends at 09:30, cutoff is 09:30 when max_age=60)
    createMinuteBucket(root, 2026, 7, 2, 9, 29, "cam_a");
    // Keep: 09:30 bucket (ends at 09:31)
    createMinuteBucket(root, 2026, 7, 2, 9, 30, "cam_b");

    const auto removed = gc.runOnce(kNow);
    EXPECT_GT(removed, 0u);
    EXPECT_FALSE(bucketExists(root, 2026, 7, 2, 9, 29));
    EXPECT_TRUE(bucketExists(root, 2026, 7, 2, 9, 30));
}

TEST(FrameRetentionGcTest, PrunesWholeHourWhenAllMinutesExpired) {
    const auto root = makeTempDir("hour");
    auto cfg = makeCfg(root);
    FrameRetentionGc gc(cfg);

    createMinuteBucket(root, 2026, 7, 2, 8, 0, "cam_a");
    createMinuteBucket(root, 2026, 7, 2, 8, 30, "cam_b");
    createMinuteBucket(root, 2026, 7, 2, 9, 30, "cam_c");

    const auto removed = gc.runOnce(kNow);
    EXPECT_GT(removed, 0u);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(root) / "20260702" / "08"));
    EXPECT_TRUE(bucketExists(root, 2026, 7, 2, 9, 30));
}

TEST(FrameRetentionGcTest, PrunesWholeDayWhenAllHoursExpired) {
    const auto root = makeTempDir("day");
    auto cfg = makeCfg(root);
    cfg.retention.max_age_minutes = 2000;  // cutoff ≈ 2026-07-01 01:10 UTC
    FrameRetentionGc gc(cfg);

    createMinuteBucket(root, 2026, 6, 30, 23, 59, "cam_old");
    createMinuteBucket(root, 2026, 7, 2, 9, 30, "cam_new");

    const auto removed = gc.runOnce(kNow);
    EXPECT_GT(removed, 0u);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(root) / "20260630"));
    EXPECT_TRUE(bucketExists(root, 2026, 7, 2, 9, 30));
}

TEST(FrameRetentionGcTest, DisabledRetentionDoesNotDelete) {
    const auto root = makeTempDir("disabled");
    auto cfg = makeCfg(root, false);
    FrameRetentionGc gc(cfg);

    createMinuteBucket(root, 2026, 7, 2, 9, 29, "cam_a");
    const auto removed = gc.runOnce(kNow);
    EXPECT_EQ(removed, 0u);
    EXPECT_TRUE(bucketExists(root, 2026, 7, 2, 9, 29));
}

TEST(FrameRetentionGcTest, RemovesEmptyParentDirectories) {
    const auto root = makeTempDir("empty_parents");
    auto cfg = makeCfg(root);
    FrameRetentionGc gc(cfg);

    createMinuteBucket(root, 2026, 7, 2, 9, 29, "cam_a");
    (void)gc.runOnce(kNow);

    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(root) / "20260702" / "09" / "29"));
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(root) / "20260702" / "09"));
}

} // namespace infer
