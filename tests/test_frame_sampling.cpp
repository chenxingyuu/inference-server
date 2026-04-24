#include <gtest/gtest.h>

#include "common/Config.h"
#include "stream/FFmpegDecoder.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace infer;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static StreamConfig makeFileCfg(int sample_fps,
                                SamplingMode mode = SamplingMode::FrameCount,
                                int duration_sec = 2) {
    StreamConfig cfg;
    cfg.id             = "test_sampling";
    cfg.url            = "lavfi:testsrc=duration=" + std::to_string(duration_sec)
                         + ":size=640x480:rate=25";
    cfg.source_type    = SourceType::File;
    cfg.loop_file      = false;
    cfg.sample_fps     = sample_fps;
    cfg.sampling_mode  = mode;
    cfg.open_timeout_ms = 5000;
    cfg.read_timeout_ms = 5000;
    cfg.use_hwdec      = false;
    return cfg;
}

// Wait for the decode thread to set running()=true, then poll until it stops.
static void waitForCompletion(FFmpegDecoder& dec,
                              std::chrono::seconds timeout = std::chrono::seconds(20)) {
    // Give the thread up to 2s to start.
    const auto start_dl = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!dec.running() && std::chrono::steady_clock::now() < start_dl) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // Poll until done.
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (dec.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ─── Enum / Config unit tests (no FFmpeg) ────────────────────────────────────

TEST(SamplingModeEnum, ValuesAreDistinct) {
    EXPECT_NE(SamplingMode::FrameCount, SamplingMode::TimeBased);
}

TEST(SamplingModeConfig, StreamConfigDefaultsToFrameCount) {
    StreamConfig cfg;
    EXPECT_EQ(cfg.sampling_mode, SamplingMode::FrameCount);
}

// ─── computeSampleInterval() unit tests ──────────────────────────────────────

TEST(ComputeSampleInterval, ValidFpsUsed) {
    EXPECT_EQ(computeSampleInterval(30.0, 5), 6);
}

TEST(ComputeSampleInterval, ZeroFpsFallsBackTo25) {
    EXPECT_EQ(computeSampleInterval(0.0, 5), 5);
}

TEST(ComputeSampleInterval, NegativeFpsFallsBackTo25) {
    EXPECT_EQ(computeSampleInterval(-1.0, 5), 5);
}

TEST(ComputeSampleInterval, ResultAtLeastOne) {
    EXPECT_EQ(computeSampleInterval(1.0, 100), 1);
}

TEST(ComputeSampleInterval, ExactMatchReturnsOne) {
    EXPECT_EQ(computeSampleInterval(25.0, 25), 1);
}

// ─── SamplingParams::shouldEmit() unit tests ─────────────────────────────────

TEST(ShouldEmit, FrameCountEmitsOnInterval) {
    SamplingParams p;
    p.mode            = SamplingMode::FrameCount;
    p.sample_interval = 5;

    EXPECT_TRUE(p.shouldEmit(std::nullopt, 0));
    EXPECT_FALSE(p.shouldEmit(std::nullopt, 1));
    EXPECT_FALSE(p.shouldEmit(std::nullopt, 2));
    EXPECT_FALSE(p.shouldEmit(std::nullopt, 4));
    EXPECT_TRUE(p.shouldEmit(std::nullopt, 5));
    EXPECT_TRUE(p.shouldEmit(std::nullopt, 10));
}

TEST(ShouldEmit, TimeBasedEmitsOnMinGap) {
    SamplingParams p;
    p.mode       = SamplingMode::TimeBased;
    p.sample_fps = 5;  // min_gap = 0.2s

    EXPECT_TRUE(p.shouldEmit(0.0, 0));    // first frame — always emit
    EXPECT_FALSE(p.shouldEmit(0.1, 1));   // 0.1s gap < 0.2s
    EXPECT_TRUE(p.shouldEmit(0.2, 2));    // exactly 0.2s gap
    EXPECT_FALSE(p.shouldEmit(0.3, 3));   // 0.1s since last emit
    EXPECT_TRUE(p.shouldEmit(0.4, 4));    // 0.2s since last emit
}

TEST(ShouldEmit, TimeBasedFallsBackToFrameCountOnNullopt) {
    SamplingParams p;
    p.mode            = SamplingMode::TimeBased;
    p.sample_fps      = 5;
    p.sample_interval = 5;  // fallback interval

    // When pts is nullopt, should behave like FrameCount with sample_interval
    EXPECT_TRUE(p.shouldEmit(std::nullopt, 0));
    EXPECT_FALSE(p.shouldEmit(std::nullopt, 1));
    EXPECT_TRUE(p.shouldEmit(std::nullopt, 5));
}

TEST(ShouldEmit, TimeBasedDoesNotEmitTwiceAtSamePts) {
    SamplingParams p;
    p.mode       = SamplingMode::TimeBased;
    p.sample_fps = 5;

    EXPECT_TRUE(p.shouldEmit(1.0, 0));
    EXPECT_FALSE(p.shouldEmit(1.0, 1));  // same timestamp
}

// ─── Integration tests (require FFmpeg/lavfi) ────────────────────────────────

TEST(FrameCountSampling, FivePerSecondFrom25fps) {
    FFmpegDecoder dec;
    std::atomic<int> count{0};

    dec.start(makeFileCfg(5, SamplingMode::FrameCount, 2), [&](Frame) { ++count; });
    waitForCompletion(dec);

    // 2s at 25fps = 50 frames; interval=5 → ~10 emitted
    EXPECT_GE(count.load(), 9);
    EXPECT_LE(count.load(), 11);
}

TEST(FrameCountSampling, FullRateWhenSampleFpsEqualsSource) {
    FFmpegDecoder dec;
    std::atomic<int> count{0};

    dec.start(makeFileCfg(25, SamplingMode::FrameCount, 1), [&](Frame) { ++count; });
    waitForCompletion(dec);

    // 1s at 25fps, interval=1 → all frames
    EXPECT_GE(count.load(), 23);
    EXPECT_LE(count.load(), 27);
}

TEST(TimeBasedSampling, FivePerSecondFrom25fps) {
    FFmpegDecoder dec;
    std::atomic<int> count{0};

    dec.start(makeFileCfg(5, SamplingMode::TimeBased, 2), [&](Frame) { ++count; });
    waitForCompletion(dec);

    // 2s stream, one frame every 0.2s → ~10 emitted
    EXPECT_GE(count.load(), 9);
    EXPECT_LE(count.load(), 11);
}

TEST(TimeBasedSampling, CappedBySourceRate) {
    FFmpegDecoder dec;
    std::atomic<int> count{0};

    // sample_fps much higher than source → emit every frame
    dec.start(makeFileCfg(100, SamplingMode::TimeBased, 1), [&](Frame) { ++count; });
    waitForCompletion(dec);

    EXPECT_GE(count.load(), 23);
    EXPECT_LE(count.load(), 27);
}

TEST(TimeBasedSampling, IndependentAcrossInstances) {
    auto run = [](SamplingMode mode) {
        FFmpegDecoder dec;
        std::atomic<int> count{0};
        dec.start(makeFileCfg(5, mode, 2), [&](Frame) { ++count; });
        waitForCompletion(dec);
        return count.load();
    };

    int a = run(SamplingMode::TimeBased);
    int b = run(SamplingMode::TimeBased);
    EXPECT_LE(std::abs(a - b), 1) << "a=" << a << " b=" << b;
}
