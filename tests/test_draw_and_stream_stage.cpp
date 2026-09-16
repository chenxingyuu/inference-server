#include <gtest/gtest.h>

#include "pipeline/stages/DetectionOverlay.h"
#include "pipeline/stages/DrawAndStreamStage.h"

#include <atomic>
#include <thread>

namespace infer {
namespace {

class FakeStreamWriter final : public IStreamWriter {
public:
    bool open(const std::string&, const std::string&, double, int gop, int, int, int) override {
        open_calls++;
        last_gop = gop;
        opened = true;
        return true;
    }

    bool write(const cv::Mat& frame) override {
        if (!opened) return false;
        if (writes_fail.load()) return false; // simulate ffmpeg spawned but sink unreachable
        write_calls++;
        last_frame = frame.clone();
        return true;
    }

    void close() override { opened = false; }
    bool isOpened() const override { return opened; }

    std::atomic<int> open_calls{0};
    std::atomic<int> write_calls{0};
    std::atomic<int> last_gop{0};
    std::atomic<bool> writes_fail{false};
    cv::Mat last_frame;
    bool opened{false};
};

TEST(MapDetectionsFromModelToFrame, ScalesToSourceResolution) {
    std::vector<Detection> dets;
    Detection d;
    d.confidence = 1.f;
    d.bbox = {160.f, 80.f, 320.f, 160.f};
    dets.push_back(d);
    InferShape shape{};
    shape.width  = 640;
    shape.height = 640;
    mapDetectionsFromModelToFrame(dets, 1920, 1080, shape);
    EXPECT_FLOAT_EQ(dets[0].bbox.x1, 480.f);
    EXPECT_FLOAT_EQ(dets[0].bbox.y1, 135.f);
    EXPECT_FLOAT_EQ(dets[0].bbox.x2, 960.f);
    EXPECT_FLOAT_EQ(dets[0].bbox.y2, 270.f);
}

TEST(DrawAndStreamStage, DrawsAndWritesFrame) {
    DrawAndStreamConfig cfg;
    cfg.output_url = "rtsp://localhost/live/test";
    cfg.protocol = "rtsp";
    cfg.queue_capacity = 2;
    cfg.gop = 12;

    auto writer = std::make_unique<FakeStreamWriter>();
    auto* writer_ptr = writer.get();
    DrawAndStreamStage stage("draw_stream", cfg, std::move(writer));
    stage.start();

    EventEnvelope in;
    in.stream_id = "cam_01";
    in.frame = std::make_shared<Frame>();
    in.frame->image = cv::Mat::zeros(80, 80, CV_8UC3);

    InferResult result;
    result.stream_id = "cam_01";
    Detection d;
    d.class_name = "car";
    d.confidence = 0.9f;
    d.bbox = BBox{10.0f, 10.0f, 40.0f, 40.0f};
    result.detections.push_back(d);
    in.infer_result = result;

    bool emitted = false;
    stage.process(in, [&](const EventEnvelope&) { emitted = true; });

    for (int i = 0; i < 20 && writer_ptr->write_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stage.stop();

    ASSERT_TRUE(emitted);
    ASSERT_GT(writer_ptr->open_calls.load(), 0);
    EXPECT_EQ(writer_ptr->last_gop.load(), 12);
    ASSERT_GT(writer_ptr->write_calls.load(), 0);
    ASSERT_FALSE(writer_ptr->last_frame.empty());
    const auto pixel = writer_ptr->last_frame.at<cv::Vec3b>(10, 10);
    EXPECT_GT(pixel[1], 0);
}

TEST(DrawAndStreamStage, AutoGopUsesRoundedFps) {
    DrawAndStreamConfig cfg;
    cfg.output_url = "rtsp://localhost/live/test";
    cfg.protocol = "rtsp";
    cfg.queue_capacity = 2;
    cfg.fps = 24.6;
    cfg.gop = 0; // auto

    auto writer = std::make_unique<FakeStreamWriter>();
    auto* writer_ptr = writer.get();
    DrawAndStreamStage stage("draw_stream_auto_gop", cfg, std::move(writer));
    stage.start();

    EventEnvelope in;
    in.stream_id = "cam_01";
    in.frame = std::make_shared<Frame>();
    in.frame->image = cv::Mat::zeros(8, 8, CV_8UC3);

    stage.process(in, [](const EventEnvelope&) {});

    for (int i = 0; i < 20 && writer_ptr->open_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stage.stop();

    ASSERT_GT(writer_ptr->open_calls.load(), 0);
    EXPECT_EQ(writer_ptr->last_gop.load(), 25);
}

// Regression: when the sink is unreachable, ffmpeg's popen() still "succeeds"
// (the child process spawns) and only the subsequent write fails. The pre-fix
// code reset the reconnect delay on every open success, so exponential backoff
// never accumulated and the stage hammered the dead sink ~once per initial
// interval, flooding the log. With the fix, backoff must escalate.
TEST(DrawAndStreamStage, BackoffEscalatesWhenWritesKeepFailing) {
    DrawAndStreamConfig cfg;
    cfg.output_url = "rtsp://localhost/live/test";
    cfg.protocol = "rtsp";
    cfg.queue_capacity = 2;
    cfg.reconnect_initial_ms = 20;
    cfg.reconnect_max_ms = 5000;

    auto writer = std::make_unique<FakeStreamWriter>();
    auto* w = writer.get();
    w->writes_fail.store(true);
    DrawAndStreamStage stage("draw_stream_backoff", cfg, std::move(writer));
    stage.start();

    EventEnvelope in;
    in.stream_id = "cam_01";
    in.frame = std::make_shared<Frame>();
    in.frame->image = cv::Mat::zeros(8, 8, CV_8UC3);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        stage.process(in, [](const EventEnvelope&) {});
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    stage.stop();

    const int opens = w->open_calls.load();
    // Exponential backoff (20,40,80,160,320ms...) => ~5 reopen attempts in 500ms.
    // The pre-fix bug produced ~25 (500ms / 20ms).
    EXPECT_GE(opens, 3);
    EXPECT_LE(opens, 9);
}

// Once writes start succeeding, an open stream must not be needlessly reopened,
// and the reconnect delay is reset so a later blip starts backoff from scratch.
TEST(DrawAndStreamStage, StopsReopeningAfterWritesSucceed) {
    DrawAndStreamConfig cfg;
    cfg.output_url = "rtsp://localhost/live/test";
    cfg.protocol = "rtsp";
    cfg.queue_capacity = 2;
    cfg.reconnect_initial_ms = 20;
    cfg.reconnect_max_ms = 5000;

    auto writer = std::make_unique<FakeStreamWriter>();
    auto* w = writer.get();
    w->writes_fail.store(true);
    DrawAndStreamStage stage("draw_stream_recover", cfg, std::move(writer));
    stage.start();

    EventEnvelope in;
    in.stream_id = "cam_01";
    in.frame = std::make_shared<Frame>();
    in.frame->image = cv::Mat::zeros(8, 8, CV_8UC3);

    auto feed_for = [&](std::chrono::milliseconds dur) {
        const auto until = std::chrono::steady_clock::now() + dur;
        while (std::chrono::steady_clock::now() < until) {
            stage.process(in, [](const EventEnvelope&) {});
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    };

    feed_for(std::chrono::milliseconds(120)); // a few failed reconnects
    w->writes_fail.store(false);              // sink comes back up
    feed_for(std::chrono::milliseconds(60));  // let it reconnect + write
    const int opens_after_recovery = w->open_calls.load();
    feed_for(std::chrono::milliseconds(200)); // keep streaming while healthy
    stage.stop();

    EXPECT_GT(w->write_calls.load(), 0);
    // A healthy, open stream should not be reopened while writes keep succeeding.
    EXPECT_LE(w->open_calls.load() - opens_after_recovery, 1);
}

} // namespace
} // namespace infer
