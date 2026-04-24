#include "pipeline/stages/ArchiveRawStage.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>

namespace infer {
namespace {

std::string makeTempDir(const std::string& suffix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path p =
        std::filesystem::temp_directory_path() / ("archive_stage_test_" + suffix + "_" + std::to_string(now));
    std::filesystem::create_directories(p);
    return p.string();
}

Frame makeFrame(const std::string& stream_id, uint64_t frame_seq) {
    Frame f;
    f.meta.stream_id = stream_id;
    f.meta.frame_seq = frame_seq;
    f.meta.capture_ts = 1777025798.0;
    f.meta.orig_width = 8;
    f.meta.orig_height = 8;
    return f;
}

} // namespace

TEST(ArchiveRawStageTest, CpuFrameWritesArchiveInfoLocalPath) {
    FrameArchiveConfig cfg;
    cfg.enabled = true;
    cfg.local_dir = makeTempDir("cpu");
    cfg.save_interval = 1;
    cfg.jpeg_quality = 90;
    cfg.queue_capacity = 16;

    auto archiver = std::make_shared<FrameArchiver>(cfg);
    ArchiveRawStage stage("archive_stage", archiver, true);

    auto frame = std::make_shared<Frame>(makeFrame("cam_cpu", 1));
    frame->is_gpu = false;
    frame->image = cv::Mat::zeros(8, 8, CV_8UC3);

    EventEnvelope in;
    in.frame = frame;

    EventEnvelope out;
    stage.process(in, [&](EventEnvelope e) { out = std::move(e); });

    ASSERT_TRUE(out.archive_info.has_value());
    EXPECT_FALSE(out.archive_info->local_path.empty());
}

TEST(ArchiveRawStageTest, GpuFrameWithFallbackImageShouldStillArchive) {
    FrameArchiveConfig cfg;
    cfg.enabled = true;
    cfg.local_dir = makeTempDir("gpu_fallback");
    cfg.save_interval = 1;
    cfg.jpeg_quality = 90;
    cfg.queue_capacity = 16;

    auto archiver = std::make_shared<FrameArchiver>(cfg);
    ArchiveRawStage stage("archive_stage", archiver, true);

    auto frame = std::make_shared<Frame>(makeFrame("cam_gpu", 2));
    frame->is_gpu = true;
    frame->image = cv::Mat::zeros(8, 8, CV_8UC3);

    EventEnvelope in;
    in.frame = frame;

    EventEnvelope out;
    stage.process(in, [&](EventEnvelope e) { out = std::move(e); });

    ASSERT_TRUE(out.archive_info.has_value());
    EXPECT_FALSE(out.archive_info->local_path.empty());
}

} // namespace infer
