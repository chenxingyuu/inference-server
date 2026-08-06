#include "archive/FrameArchiver.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <thread>

namespace infer {
namespace {

std::string makeTempDir(const std::string& suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path p =
        std::filesystem::temp_directory_path() / ("frame_archiver_test_" + suffix + "_" + std::to_string(now));
    std::filesystem::create_directories(p);
    return p.string();
}

StreamMeta makeMeta(const std::string& stream_id, uint64_t frame_seq) {
    StreamMeta meta;
    meta.stream_id = stream_id;
    meta.frame_seq = frame_seq;
    meta.capture_ts = 1777025798.0;
    meta.orig_width = 16;
    meta.orig_height = 16;
    return meta;
}

FrameArchiveConfig makeCfg(const std::string& local_dir) {
    FrameArchiveConfig cfg;
    cfg.enabled = true;
    cfg.local_dir = local_dir;
    cfg.save_interval = 1;
    cfg.jpeg_quality = 90;
    cfg.queue_capacity = 16;
    cfg.worker_count = 1;
    return cfg;
}

bool waitForFile(const std::filesystem::path& path, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return std::filesystem::exists(path);
}

} // namespace

TEST(FrameArchiverTest, SubmitStillReturnsPathBeforeWrite) {
    const std::string local_dir = makeTempDir("early_return");
    FrameArchiver archiver(makeCfg(local_dir));

    cv::Mat frame = cv::Mat::zeros(16, 16, CV_8UC3);
    const auto result = archiver.submit(makeMeta("cam_early", 1), &frame);

    EXPECT_EQ(result.upload_state, "queued");
    EXPECT_FALSE(result.local_path.empty());
    EXPECT_FALSE(result.frame_url.empty());
    EXPECT_EQ(result.frame_url, "20260424/10/16/cam_early/1777025798000_1.jpg");
    EXPECT_TRUE(result.local_path.find("cam_early") != std::string::npos);
}

TEST(FrameArchiverTest, SubmitAsyncWriteProducesReadableJpeg) {
    const std::string local_dir = makeTempDir("readable");
    FrameArchiver archiver(makeCfg(local_dir));

    cv::Mat frame = cv::Mat::zeros(16, 16, CV_8UC3);
    frame.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    const auto result = archiver.submit(makeMeta("cam_read", 2), &frame);
    ASSERT_EQ(result.upload_state, "queued");

    const std::filesystem::path final_path(result.local_path);
    ASSERT_TRUE(waitForFile(final_path, std::chrono::seconds(2)))
        << "expected archived file at " << final_path;

    const cv::Mat decoded = cv::imread(final_path.string(), cv::IMREAD_COLOR);
    ASSERT_FALSE(decoded.empty());
    EXPECT_EQ(decoded.cols, 16);
    EXPECT_EQ(decoded.rows, 16);
}

TEST(FrameArchiverTest, AtomicWriteLeavesNoTmpFile) {
    const std::string local_dir = makeTempDir("no_tmp");
    FrameArchiver archiver(makeCfg(local_dir));

    cv::Mat frame = cv::Mat::zeros(8, 8, CV_8UC3);
    const auto result = archiver.submit(makeMeta("cam_tmp", 3), &frame);
    ASSERT_EQ(result.upload_state, "queued");

    const std::filesystem::path final_path(result.local_path);
    const std::filesystem::path tmp_path =
        final_path.parent_path() / (".tmp_" + final_path.filename().string());
    ASSERT_TRUE(waitForFile(final_path, std::chrono::seconds(2)));

    EXPECT_FALSE(std::filesystem::exists(tmp_path)) << "orphan tmp file: " << tmp_path;
}

TEST(FrameArchiverTest, QueueFullClearsFrameUrlAndLocalPath) {
    // worker_count=0: nothing drains the queue, so we can deterministically fill it.
    FrameArchiveConfig cfg = makeCfg(makeTempDir("queue_full"));
    cfg.queue_capacity = 2;
    cfg.worker_count = 0;
    cfg.public_base_url = "http://frame-nginx:8082/frames";
    FrameArchiver archiver(cfg);

    cv::Mat frame = cv::Mat::zeros(8, 8, CV_8UC3);
    ASSERT_EQ(archiver.submit(makeMeta("cam_qfull", 1), &frame).upload_state, "queued");
    ASSERT_EQ(archiver.submit(makeMeta("cam_qfull", 2), &frame).upload_state, "queued");

    const auto dropped = archiver.submit(makeMeta("cam_qfull", 3), &frame);
    EXPECT_EQ(dropped.upload_state, "failed");
    EXPECT_TRUE(dropped.frame_url.empty()) << "must not advertise a URL for a frame that was not archived";
    EXPECT_TRUE(dropped.local_path.empty());
    EXPECT_TRUE(dropped.object_key.empty());
}

} // namespace infer
