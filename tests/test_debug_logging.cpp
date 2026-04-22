// Reproducer tests for debug-level log points added to hot-path stages.
// Each test is RED before the LOG_DEBUG call is added; GREEN after.
//
// Coverage:
//   YOLOv8Decoder::decode()      – emits per-batch det count at DEBUG
//   ByteTrackTracker::update()   – emits active track count at DEBUG

#include <gtest/gtest.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/log_msg.h>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>

#include "common/Logger.h"
#include "decoder/YOLOv8Decoder.h"
#include "tracker/ByteTrackTracker.h"

using namespace infer;

// ── Capture sink ──────────────────────────────────────────────────────────────

struct LogRecord {
    spdlog::level::level_enum level;
    std::string               msg;
};

class CaptureSink : public spdlog::sinks::base_sink<std::mutex> {
public:
    std::vector<LogRecord> records;

    bool hasLevelWith(spdlog::level::level_enum lvl, const std::string& sub) const {
        return std::any_of(records.begin(), records.end(), [&](const auto& r) {
            return r.level == lvl && r.msg.find(sub) != std::string::npos;
        });
    }

protected:
    void sink_it_(const spdlog::details::log_msg& m) override {
        records.push_back({m.level, std::string(m.payload.begin(), m.payload.end())});
    }
    void flush_() override {}
};

class LogCapture {
public:
    explicit LogCapture(const std::string& level = "debug") {
        original_ = spdlog::default_logger();
        sink_     = std::make_shared<CaptureSink>();
        auto logger = std::make_shared<spdlog::logger>("capture", sink_);
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::from_str(level));
    }
    ~LogCapture() { spdlog::set_default_logger(original_); }
    CaptureSink& sink() { return *sink_; }

private:
    std::shared_ptr<spdlog::logger> original_;
    std::shared_ptr<CaptureSink>    sink_;
};

// ── YOLOv8Decoder ─────────────────────────────────────────────────────────────

TEST(DebugLog, YOLOv8DecodeEmitsDetCount) {
    LogCapture capture("debug");

    YOLOv8Decoder dec(1); // 1 class

    // Minimal output buffer: [4+1 rows] × 8400 anchors.
    // Anchor 0 gets high class-0 confidence + valid bbox.
    std::vector<float> out(5 * 8400, 0.0f);
    out[4 * 8400 + 0] = 0.9f;   // class-0 probability
    out[0 * 8400 + 0] = 320.f;  // cx
    out[1 * 8400 + 0] = 320.f;  // cy
    out[2 * 8400 + 0] = 100.f;  // w
    out[3 * 8400 + 0] = 100.f;  // h

    InferShape shape{640, 640};
    auto results = dec.decode(out.data(), 1, shape, 0.5f, 0.45f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), 1u);

    EXPECT_TRUE(capture.sink().hasLevelWith(spdlog::level::debug, "dets"))
        << "YOLOv8Decoder::decode() should emit a DEBUG line containing 'dets'";
}

TEST(DebugLog, YOLOv8DecodeEmitsZeroDetCount) {
    LogCapture capture("debug");

    YOLOv8Decoder dec(1);
    // All-zero output → no detections
    std::vector<float> out(5 * 8400, 0.0f);
    InferShape shape{640, 640};
    dec.decode(out.data(), 1, shape, 0.5f, 0.45f);

    EXPECT_TRUE(capture.sink().hasLevelWith(spdlog::level::debug, "dets=0"))
        << "YOLOv8Decoder::decode() should report dets=0 on empty output";
}

// ── ByteTrackTracker ──────────────────────────────────────────────────────────

TEST(DebugLog, ByteTrackUpdateEmitsActiveTracks) {
    LogCapture capture("debug");

    ByteTrackConfig cfg;
    cfg.high_det_thresh    = 0.5f;
    cfg.low_det_thresh     = 0.1f;
    cfg.match_iou_thresh   = 0.3f;
    cfg.min_hits_to_confirm = 1;
    cfg.max_lost_frames    = 30;
    ByteTrackTracker tracker(cfg);

    Detection d;
    d.confidence = 0.9f;
    d.bbox       = {10.f, 10.f, 50.f, 50.f};
    d.class_id   = 0;
    std::vector<Detection> dets = {d};

    tracker.update(1, dets);

    EXPECT_TRUE(capture.sink().hasLevelWith(spdlog::level::debug, "active"))
        << "ByteTrackTracker::update() should emit a DEBUG line containing 'active'";
}

TEST(DebugLog, ByteTrackUpdateEmitsZeroActiveTracks) {
    LogCapture capture("debug");

    ByteTrackConfig cfg;
    cfg.high_det_thresh    = 0.5f;
    cfg.low_det_thresh     = 0.1f;
    cfg.match_iou_thresh   = 0.3f;
    cfg.min_hits_to_confirm = 3; // require 3 hits to confirm
    cfg.max_lost_frames    = 30;
    ByteTrackTracker tracker(cfg);

    std::vector<Detection> dets; // no detections
    tracker.update(1, dets);

    EXPECT_TRUE(capture.sink().hasLevelWith(spdlog::level::debug, "active=0"))
        << "ByteTrackTracker::update() should report active=0 on empty input";
}
