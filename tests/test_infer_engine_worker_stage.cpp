#include <gtest/gtest.h>

#include "decoder/DecoderFactory.h"
#include "infer/IInferBackend.h"
#include "pipeline/stages/InferEngineWorkerStage.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace infer {
namespace {

class FakeBackend final : public IInferBackend {
public:
    explicit FakeBackend(std::shared_ptr<std::vector<int>> loaded_devices)
        : devices_(std::move(loaded_devices)) {}

    void loadModel(const ModelConfig& cfg) override {
        if (devices_) devices_->push_back(cfg.device_id);
    }

    void infer(const Batch& batch, std::vector<float>& output) override {
        const int nc = 80;
        const int na = 8400;
        const int rows = 4 + nc;
        output.assign(static_cast<std::size_t>(batch.size()) * static_cast<std::size_t>(rows) * static_cast<std::size_t>(na),
                      0.001f);
    }

    void unloadModel() override {}

    int        maxBatchSize() const override { return 16; }
    DeviceType deviceType() const override { return DeviceType::CPU; }
    bool       isLoaded() const override { return true; }

private:
    std::shared_ptr<std::vector<int>> devices_;
};

ModelConfig makeModel() {
    ModelConfig mc;
    mc.id = "m_test";
    mc.version = YOLOVersion::v8;
    mc.backend = DeviceType::CPU;
    mc.onnx_path = "models/placeholder.onnx";
    mc.batch_size = 1;
    mc.instance_count = 2;
    mc.device_ids = {11, 22};
    mc.num_classes = 80;
    mc.input_shape.batch = 1;
    mc.input_shape.channels = 3;
    mc.input_shape.height = 640;
    mc.input_shape.width = 640;
    mc.max_queue_delay_us = 5000;
    return mc;
}

} // namespace

TEST(InferEngineWorkerStage, LoadsEachInstanceOnDistinctDeviceIds) {
    auto devices = std::make_shared<std::vector<int>>();

    ModelConfig mc = makeModel();

    auto backend_factory = [devices](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(std::make_unique<FakeBackend>(devices));
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage("infer_1", mc, backend_factory, decoder_factory);
    ASSERT_EQ(stage.workerInstanceCount(), 2);

    stage.start();
    EXPECT_EQ(devices->size(), 2u);
    EXPECT_EQ((*devices)[0], 11);
    EXPECT_EQ((*devices)[1], 22);

    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> emit_count{0};

    EventEnvelope ev;
    ev.stream_id = "cam_a";
    ev.frame_seq = 0;
    ev.frame = std::make_shared<Frame>();
    ev.frame->is_gpu = false;
    ev.frame->image = cv::Mat::zeros(64, 64, CV_8UC3);
    ev.frame->meta.stream_id = "cam_a";
    ev.frame->meta.frame_seq = 0;
    ev.frame->meta.capture_ts = 1.0;
    ev.frame->meta.capture_mono_ns = 1000;

    stage.process(ev, [&](const EventEnvelope& out) {
        ASSERT_TRUE(out.infer_result.has_value());
        EXPECT_EQ(out.infer_result->stream_id, "cam_a");
        emit_count.fetch_add(1, std::memory_order_relaxed);
        cv.notify_all();
    });

    {
        std::unique_lock<std::mutex> lk(mu);
        ASSERT_TRUE(cv.wait_for(lk, std::chrono::seconds(2), [&] {
            return emit_count.load(std::memory_order_relaxed) >= 1;
        }));
    }

    stage.stop();
}

} // namespace infer
