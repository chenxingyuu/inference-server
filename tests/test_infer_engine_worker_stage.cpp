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

// BUG-1 regression: each stream's result must be emitted via its own emit fn.
// With batch_size=2, both frames accumulate before inline flush; the old code
// stamped all inflight entries with the *last* emit (emit_b), so cam_a results
// arrived through emit_b.  After the fix, each PendingEmit carries its own fn.
TEST(InferEngineWorkerStage, MultiStreamEmitRoutedCorrectly) {
    ModelConfig mc = makeModel();
    mc.batch_size    = 2;
    mc.instance_count = 1;
    mc.device_ids     = {0};

    auto backend_factory = [](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(
            std::make_unique<FakeBackend>(nullptr));
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage("infer_ms", mc, backend_factory, decoder_factory);
    stage.start();

    std::mutex              mu_a, mu_b;
    std::condition_variable cv_a, cv_b;
    std::atomic<int>        a_count{0}, b_count{0};
    std::atomic<bool>       a_wrong{false}, b_wrong{false};

    auto make_frame = [](const std::string& sid, uint64_t seq) {
        EventEnvelope ev;
        ev.stream_id       = sid;
        ev.frame_seq       = seq;
        ev.frame           = std::make_shared<Frame>();
        ev.frame->is_gpu   = false;
        ev.frame->image    = cv::Mat::zeros(64, 64, CV_8UC3);
        ev.frame->meta.stream_id     = sid;
        ev.frame->meta.frame_seq     = seq;
        ev.frame->meta.capture_ts    = 1.0;
        ev.frame->meta.capture_mono_ns = 1000;
        return ev;
    };

    EventEnvelope ev_a = make_frame("cam_a", 10);
    EventEnvelope ev_b = make_frame("cam_b", 20);

    stage.process(ev_a, [&](const EventEnvelope& out) {
        if (!out.infer_result || out.infer_result->stream_id != "cam_a") a_wrong.store(true);
        a_count.fetch_add(1, std::memory_order_relaxed);
        cv_a.notify_all();
    });
    stage.process(ev_b, [&](const EventEnvelope& out) {
        if (!out.infer_result || out.infer_result->stream_id != "cam_b") b_wrong.store(true);
        b_count.fetch_add(1, std::memory_order_relaxed);
        cv_b.notify_all();
    });

    {
        std::unique_lock<std::mutex> lk(mu_a);
        ASSERT_TRUE(cv_a.wait_for(lk, std::chrono::seconds(3),
                                   [&] { return a_count.load() >= 1; }))
            << "cam_a result never arrived";
    }
    {
        std::unique_lock<std::mutex> lk(mu_b);
        ASSERT_TRUE(cv_b.wait_for(lk, std::chrono::seconds(3),
                                   [&] { return b_count.load() >= 1; }))
            << "cam_b result never arrived";
    }

    EXPECT_FALSE(a_wrong.load()) << "cam_a result routed through wrong emit fn";
    EXPECT_FALSE(b_wrong.load()) << "cam_b result routed through wrong emit fn";

    stage.stop();
}

// BUG-2 regression: explicit stop() + destructor stop() must not crash.
TEST(InferEngineWorkerStage, DoubleStopDoesNotCrash) {
    ModelConfig mc = makeModel();
    mc.instance_count = 1;
    mc.device_ids     = {0};

    auto backend_factory = [](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(
            std::make_unique<FakeBackend>(nullptr));
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage("infer_ds", mc, backend_factory, decoder_factory);
    stage.start();
    stage.stop();
    EXPECT_NO_FATAL_FAILURE(stage.stop()); // second explicit stop
    // destructor calls stop() a third time — must also be safe
}

// BUG-2 variant: stop() before start() must not crash.
TEST(InferEngineWorkerStage, StopBeforeStartDoesNotCrash) {
    ModelConfig mc = makeModel();
    mc.instance_count = 1;
    mc.device_ids     = {0};

    auto backend_factory = [](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(
            std::make_unique<FakeBackend>(nullptr));
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage("infer_sbs", mc, backend_factory, decoder_factory);
    EXPECT_NO_FATAL_FAILURE(stage.stop()); // stop before start
}

} // namespace infer
