#include <gtest/gtest.h>

#include "pipeline/stages/DrawAndStreamStage.h"

#include <atomic>
#include <thread>

namespace infer {
namespace {

class FakeStreamWriter final : public IStreamWriter {
public:
    bool open(const std::string&, const std::string&, double, int, int, int) override {
        open_calls++;
        opened = true;
        return true;
    }

    bool write(const cv::Mat& frame) override {
        if (!opened) return false;
        write_calls++;
        last_frame = frame.clone();
        return true;
    }

    void close() override { opened = false; }
    bool isOpened() const override { return opened; }

    std::atomic<int> open_calls{0};
    std::atomic<int> write_calls{0};
    cv::Mat last_frame;
    bool opened{false};
};

TEST(DrawAndStreamStage, DrawsAndWritesFrame) {
    DrawAndStreamConfig cfg;
    cfg.output_url = "rtsp://localhost/live/test";
    cfg.protocol = "rtsp";
    cfg.queue_capacity = 2;

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
    ASSERT_GT(writer_ptr->write_calls.load(), 0);
    ASSERT_FALSE(writer_ptr->last_frame.empty());
    const auto pixel = writer_ptr->last_frame.at<cv::Vec3b>(10, 10);
    EXPECT_GT(pixel[1], 0);
}

} // namespace
} // namespace infer
