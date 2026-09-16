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
#include <stdexcept>
#include <thread>
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

// Backend whose infer() always throws, faithfully reproducing an inference
// exception in InferWorker (caught at InferWorker.cpp: the whole batch is
// dropped and no result is published). Frames handed to this backend become
// orphaned inflight entries with no result ever returning.
class ThrowingBackend final : public IInferBackend {
public:
    void loadModel(const ModelConfig&) override {}
    void infer(const Batch&, std::vector<float>&) override {
        throw std::runtime_error("simulated inference failure");
    }
    void unloadModel() override {}
    int        maxBatchSize() const override { return 16; }
    DeviceType deviceType() const override { return DeviceType::CPU; }
    bool       isLoaded() const override { return true; }
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

// Emits one high-confidence class-0 box in YOLOv8 [4+nc, na] layout.
class DetectingBackend final : public IInferBackend {
public:
    void loadModel(const ModelConfig&) override {}
    void infer(const Batch& batch, std::vector<float>& output) override {
        const int nc = 80;
        const int na = 8400;
        const int rows = 4 + nc;
        output.assign(static_cast<std::size_t>(batch.size()) * rows * na, 0.f);
        for (int b = 0; b < batch.size(); ++b) {
            float* data = output.data() + static_cast<std::size_t>(b) * rows * na;
            const int a = 0;
            data[0 * na + a] = 320.f;
            data[1 * na + a] = 320.f;
            data[2 * na + a] = 100.f;
            data[3 * na + a] = 100.f;
            data[(4 + 0) * na + a] = 0.95f;
        }
    }
    void unloadModel() override {}
    int        maxBatchSize() const override { return 16; }
    DeviceType deviceType() const override { return DeviceType::CPU; }
    bool       isLoaded() const override { return true; }
};

class ClassifierBackend final : public IInferBackend {
public:
    explicit ClassifierBackend(int num_classes) : num_classes_(num_classes) {}
    void loadModel(const ModelConfig&) override {}
    void infer(const Batch& batch, std::vector<float>& output) override {
        output.assign(static_cast<std::size_t>(batch.size()) * static_cast<std::size_t>(num_classes_),
                      0.05f);
        for (int b = 0; b < batch.size(); ++b) {
            // class 1 wins
            output[static_cast<std::size_t>(b) * num_classes_ + 1] = 0.99f;
        }
    }
    void unloadModel() override {}
    int        maxBatchSize() const override { return 16; }
    DeviceType deviceType() const override { return DeviceType::CPU; }
    bool       isLoaded() const override { return true; }

private:
    int num_classes_;
};

ModelConfig makeClassifierModel() {
    ModelConfig mc;
    mc.id = "classifier_01";
    mc.model_type = ModelType::Classifier;
    mc.backend = DeviceType::CPU;
    mc.onnx_path = "models/placeholder.onnx";
    mc.batch_size = 1;
    mc.instance_count = 1;
    mc.device_ids = {0};
    mc.num_classes = 3;
    mc.class_names = {"sedan", "suv", "truck"};
    mc.conf_thresh = 0.4f;
    mc.input_shape.batch = 1;
    mc.input_shape.channels = 3;
    mc.input_shape.height = 64;
    mc.input_shape.width = 64;
    mc.max_queue_delay_us = 5000;
    return mc;
}

ModelConfig makeCascadePrimary() {
    ModelConfig mc = makeModel();
    mc.id = "detector_cascade";
    mc.instance_count = 1;
    mc.device_ids = {0};
    CascadeConfig cas;
    cas.model_id = "classifier_01";
    cas.attribute_key = "vehicle_type";
    cas.trigger_classes = {0};
    cas.crop_expand = 0.1f;
    mc.cascade.push_back(cas);
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

// LEAK regression: a frame whose inference result never arrives (dropped batch /
// inference exception) leaves a permanent entry in inflight_, each pinning a whole
// shared_ptr<Frame>. Without a stale sweep, inflight_ grows without bound and RSS
// climbs monotonically. The sweep must evict such orphans after the stale timeout.
TEST(InferEngineWorkerStage, EvictsStaleInflightWhenResultNeverArrives) {
    ModelConfig mc = makeModel();
    mc.batch_size     = 1;
    mc.instance_count = 1;
    mc.device_ids     = {0};

    auto backend_factory = [](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(std::make_unique<ThrowingBackend>());
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage("infer_leak", mc, backend_factory, decoder_factory);
    stage.setInflightStaleTimeoutForTest(std::chrono::milliseconds(100));
    stage.start();

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

    stage.process(ev, [&](const EventEnvelope&) {
        emit_count.fetch_add(1, std::memory_order_relaxed);
    });

    // The throwing backend drops the batch, so an orphaned inflight entry appears
    // and no result is ever emitted.
    bool orphan_seen = false;
    for (int i = 0; i < 50; ++i) {
        if (stage.inflightSize() == 1) { orphan_seen = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(orphan_seen) << "expected exactly one orphaned inflight entry";
    EXPECT_EQ(emit_count.load(std::memory_order_relaxed), 0)
        << "throwing backend must not emit a result";

    // After the stale timeout, the periodic sweep must evict the orphan and
    // release the frame it pins.
    bool evicted = false;
    for (int i = 0; i < 100; ++i) {  // up to ~1s
        if (stage.inflightSize() == 0) { evicted = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(evicted) << "stale inflight entry was never evicted (memory leak)";

    stage.stop();
}

TEST(InferEngineWorkerStage, CascadeMergesSecondaryAttributesIntoEmit) {
    ModelConfig primary = makeCascadePrimary();
    ModelConfig secondary = makeClassifierModel();

    auto backend_factory = [](const ModelConfig& c) -> std::unique_ptr<IInferBackend> {
        if (c.model_type == ModelType::Classifier) {
            return std::make_unique<ClassifierBackend>(c.num_classes);
        }
        return std::make_unique<DetectingBackend>();
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage(
        "infer_cascade", primary, backend_factory, decoder_factory, {secondary});
    EXPECT_EQ(stage.secondaryGroupCount(), 1);
    stage.start();

    std::mutex mu;
    std::condition_variable cv;
    std::atomic<int> emit_count{0};
    std::string attr_value;
    bool has_det = false;

    EventEnvelope ev;
    ev.stream_id = "cam_a";
    ev.frame_seq = 7;
    ev.frame = std::make_shared<Frame>();
    ev.frame->is_gpu = false;
    ev.frame->image = cv::Mat::zeros(640, 640, CV_8UC3);
    ev.frame->meta.stream_id = "cam_a";
    ev.frame->meta.frame_seq = 7;
    ev.frame->meta.capture_ts = 1.5;
    ev.frame->meta.capture_mono_ns = 1000;

    stage.process(ev, [&](const EventEnvelope& out) {
        ASSERT_TRUE(out.infer_result.has_value());
        has_det = !out.infer_result->detections.empty();
        if (has_det) {
            auto it = out.infer_result->detections[0].attributes.find("vehicle_type");
            if (it != out.infer_result->detections[0].attributes.end()) {
                attr_value = it->second;
            }
        }
        emit_count.fetch_add(1, std::memory_order_relaxed);
        cv.notify_all();
    });

    {
        std::unique_lock<std::mutex> lk(mu);
        ASSERT_TRUE(cv.wait_for(lk, std::chrono::seconds(3), [&] {
            return emit_count.load(std::memory_order_relaxed) >= 1;
        })) << "cascade emit timed out";
    }

    EXPECT_TRUE(has_det);
    EXPECT_EQ(attr_value, "suv");

    stage.stop();
}

TEST(InferEngineWorkerStage, CascadeDoubleStopDoesNotCrash) {
    ModelConfig primary = makeCascadePrimary();
    ModelConfig secondary = makeClassifierModel();

    auto backend_factory = [](const ModelConfig& c) -> std::unique_ptr<IInferBackend> {
        if (c.model_type == ModelType::Classifier) {
            return std::make_unique<ClassifierBackend>(c.num_classes);
        }
        return std::make_unique<FakeBackend>(nullptr);
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    InferEngineWorkerStage stage(
        "infer_cascade_stop", primary, backend_factory, decoder_factory, {secondary});
    stage.start();
    stage.stop();
    EXPECT_NO_FATAL_FAILURE(stage.stop());
}

TEST(InferEngineWorkerStage, RejectsMissingSecondaryModelAtConstruction) {
    ModelConfig primary = makeCascadePrimary();
    auto backend_factory = [](const ModelConfig&) {
        return std::unique_ptr<IInferBackend>(std::make_unique<FakeBackend>(nullptr));
    };
    auto decoder_factory = [](const ModelConfig& c) { return createDecoder(c); };

    EXPECT_THROW(
        InferEngineWorkerStage("infer_bad", primary, backend_factory, decoder_factory, {}),
        std::runtime_error);
}

} // namespace infer
