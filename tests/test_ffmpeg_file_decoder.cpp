#include <gtest/gtest.h>

#include "stream/FFmpegDecoder.h"
#include "common/Config.h"

#include <atomic>
#include <chrono>
#include <thread>

using namespace infer;

// Uses FFmpeg's lavfi testsrc virtual source so no real file is required.
// The testsrc generates 1 second of 640×480 video at 25 fps.
static constexpr const char* kTestSrcUrl =
    "testsrc=duration=1:size=640x480:rate=25";
static constexpr const char* kTestSrcFormat = "lavfi";

static StreamConfig makeFileCfg(bool loop = false) {
    StreamConfig cfg;
    cfg.id            = "test_file";
    // lavfi URL opened via format specifier; FFmpeg accepts this path as-is
    // when avformat_open_input is called with the right format dict.
    // We embed the format hint in the URL for the lavfi demuxer:
    cfg.url           = std::string(kTestSrcFormat) + ":" + kTestSrcUrl;
    cfg.source_type   = SourceType::File;
    cfg.loop_file     = loop;
    cfg.sample_fps    = 25;   // decode every frame
    cfg.open_timeout_ms = 5000;
    cfg.read_timeout_ms = 5000;
    cfg.use_hwdec     = false;
    return cfg;
}

// ─── EOF: decoder stops cleanly without spinning ────────────────────────────

TEST(FFmpegFileDecoder, StopsOnEOFWithoutReconnect) {
    FFmpegDecoder decoder;
    std::atomic<int> frame_count{0};

    decoder.start(makeFileCfg(/*loop=*/false), [&](Frame f) {
        ++frame_count;
        (void)f;
    });

    // 1-second clip at 25fps → expect ~25 frames, wait at most 10 s
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (decoder.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Decoder should have stopped on its own (no reconnect loop)
    EXPECT_FALSE(decoder.running()) << "Decoder should stop on EOF without reconnect";
    EXPECT_GT(frame_count.load(), 0) << "At least one frame should have been decoded";
}

// ─── Frames are produced ─────────────────────────────────────────────────────

TEST(FFmpegFileDecoder, DecodesFramesFromVirtualSource) {
    FFmpegDecoder decoder;
    std::atomic<int> frame_count{0};

    decoder.start(makeFileCfg(), [&](Frame f) {
        ++frame_count;
        EXPECT_GT(f.meta.orig_width,  0);
        EXPECT_GT(f.meta.orig_height, 0);
        EXPECT_FALSE(f.meta.stream_id.empty());
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (decoder.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_GT(frame_count.load(), 0);
}

// ─── stop() is safe to call before natural EOF ──────────────────────────────

TEST(FFmpegFileDecoder, StopBeforeEOF) {
    FFmpegDecoder decoder;
    std::atomic<int> frame_count{0};

    decoder.start(makeFileCfg(), [&](Frame) { ++frame_count; });

    // Let it run briefly then stop manually
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_NO_THROW(decoder.stop());
    EXPECT_FALSE(decoder.running());
}
