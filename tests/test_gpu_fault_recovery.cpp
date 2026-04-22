#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "pipeline/BatchPolicy.h"
#include "infer/GpuFault.h"
#include "infer/IInferBackend.h"
#include "decoder/IYOLODecoder.h"
#include "publisher/IPublisher.h"
#include "pipeline/InferWorker.h"
#include "metrics/Metrics.h"
#include <thread>
#include <chrono>
#include <future>

using namespace infer;
using namespace testing;

// ── Mocks ─────────────────────────────────────────────────────────────────

class MockBackend : public IInferBackend {
public:
    MOCK_METHOD(void, loadModel,   (const ModelConfig&), (override));
    MOCK_METHOD(void, infer,       (const Batch&, std::vector<float>&), (override));
    MOCK_METHOD(void, unloadModel, (), (override));
    MOCK_METHOD(int,        maxBatchSize, (), (const, override));
    MOCK_METHOD(DeviceType, deviceType,   (), (const, override));
    MOCK_METHOD(bool,       isLoaded,     (), (const, override));
};

class MockDecoder : public IYOLODecoder {
public:
    MOCK_METHOD((std::vector<std::vector<Detection>>), decode,
                (const float*, int, const InferShape&, float, float), (override));
    MOCK_METHOD(YOLOVersion, version, (), (const, override));
};

class MockPublisher : public IPublisher {
public:
    MOCK_METHOD(void, publish, (InferResult), (override));
    MOCK_METHOD(void, flush,   (), (override));
};

// Helper: minimal valid ModelConfig
static ModelConfig makeModelConfig(const std::string& id = "test_model") {
    ModelConfig cfg;
    cfg.id          = id;
    cfg.batch_size  = 16;
    cfg.input_shape = {640, 640};
    cfg.conf_thresh    = 0.3f;
    cfg.nms_thresh     = 0.45f;
    return cfg;
}

// Helper: make a single-frame CPU batch
static Batch makeBatch(const std::string& stream_id = "s0") {
    Batch b;
    b.is_gpu = false;
    cv::Mat frame(640, 640, CV_8UC3, cv::Scalar(0));
    b.frames.push_back(frame);
    StreamMeta meta;
    meta.stream_id = stream_id;
    meta.capture_ts = 0.0;
    meta.capture_mono_ns = 0;
    meta.frame_seq = 1;
    b.metas.push_back(meta);
    return b;
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 1: BatchPolicy — pure state machine logic
// ═══════════════════════════════════════════════════════════════════════════

TEST(BatchPolicyTest, InitialMaxIs16) {
    BatchPolicy p;
    EXPECT_EQ(p.current_max, 16);
}

TEST(BatchPolicyTest, OnOomHalvesMax) {
    BatchPolicy p;
    p.onOOM();
    EXPECT_EQ(p.current_max, 8);
    p.onOOM();
    EXPECT_EQ(p.current_max, 4);
    p.onOOM();
    EXPECT_EQ(p.current_max, 2);
    p.onOOM();
    EXPECT_EQ(p.current_max, 1);
}

TEST(BatchPolicyTest, OnOomNeverBelowOne) {
    BatchPolicy p;
    for (int i = 0; i < 10; ++i) p.onOOM();
    EXPECT_EQ(p.current_max, 1);
    EXPECT_GE(p.current_max, 1);
}

TEST(BatchPolicyTest, OnSuccessRestoresMaxAfterThreshold) {
    BatchPolicy p;
    p.onOOM(); // current_max = 8
    ASSERT_EQ(p.current_max, 8);

    for (int i = 0; i < BatchPolicy::kRestoreThreshold; ++i) {
        p.onSuccess();
    }
    EXPECT_EQ(p.current_max, 16);
}

TEST(BatchPolicyTest, IsExhaustedAfterRepeatedOomAtMin) {
    BatchPolicy p;
    // Drive to minimum
    for (int i = 0; i < 4; ++i) p.onOOM();
    ASSERT_EQ(p.current_max, 1);
    // 3 more OOMs at current_max=1 → exhausted
    p.onOOM();
    p.onOOM();
    p.onOOM();
    EXPECT_TRUE(p.isExhausted());
}

TEST(BatchPolicyTest, OnOomResetsSuccessStreak) {
    BatchPolicy p;
    for (int i = 0; i < 50; ++i) p.onSuccess();
    p.onOOM();
    EXPECT_EQ(p.success_streak, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 2: GpuFaultType classification
// ═══════════════════════════════════════════════════════════════════════════

TEST(GpuFaultTypeTest, MemAllocIsOom) {
    EXPECT_EQ(classifyGpuError(kCudaErrorMemAlloc), GpuFaultType::OOM);
}

TEST(GpuFaultTypeTest, InvalidContextIsContextLost) {
    EXPECT_EQ(classifyGpuError(kCudaErrorInvalidContext), GpuFaultType::CONTEXT_LOST);
    EXPECT_EQ(classifyGpuError(kCudaErrorInvalidDevice), GpuFaultType::CONTEXT_LOST);
}

TEST(GpuFaultTypeTest, OtherErrorIsEngineFault) {
    // Some arbitrary non-OOM, non-context error code
    EXPECT_EQ(classifyGpuError(999), GpuFaultType::ENGINE_FAULT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 3: New Prometheus metrics
// ═══════════════════════════════════════════════════════════════════════════

TEST(GpuFaultMetrics, OomCounterIncremented) {
    auto& m = Metrics::get();
    m.incGpuOom("gfm_model1");
    m.incGpuOom("gfm_model1");

    const std::string out = m.serialize();
    EXPECT_NE(out.find("gpu_oom_total"), std::string::npos);
    EXPECT_NE(out.find("gpu_oom_total{worker_id=\"gfm_model1\"} 2"), std::string::npos);
}

TEST(GpuFaultMetrics, EngineFaultCounterIncremented) {
    auto& m = Metrics::get();
    m.incGpuEngineFault("gfm_model2");

    const std::string out = m.serialize();
    EXPECT_NE(out.find("gpu_engine_fault_total"), std::string::npos);
    EXPECT_NE(out.find("gpu_engine_fault_total{worker_id=\"gfm_model2\"} 1"), std::string::npos);
}

TEST(GpuFaultMetrics, WorkerStateGaugeSet) {
    auto& m = Metrics::get();
    m.setInferWorkerState("gfm_model3", 1); // RECOVERING

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_worker_state"), std::string::npos);
    EXPECT_NE(out.find("infer_worker_state{worker_id=\"gfm_model3\"} 1"), std::string::npos);
}

TEST(GpuFaultMetrics, BatchSizeCurrentGaugeSet) {
    auto& m = Metrics::get();
    m.setInferBatchSizeCurrent("gfm_model4", 4);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_batch_size_current"), std::string::npos);
    EXPECT_NE(out.find("infer_batch_size_current{worker_id=\"gfm_model4\"} 4"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 4: InferWorker state machine with MockBackend
// ═══════════════════════════════════════════════════════════════════════════

TEST(InferWorkerRecovery, OomRequeuesAndIncrementsMetric) {
    auto cfg = makeModelConfig("iw_oom_m1");
    auto* raw_backend = new MockBackend();
    auto* raw_decoder = new MockDecoder();
    MockPublisher publisher;

    std::promise<void> oom_seen;
    auto oom_future = oom_seen.get_future();
    bool oom_triggered = false;

    EXPECT_CALL(*raw_backend, loadModel(_)).Times(AnyNumber());
    EXPECT_CALL(*raw_backend, unloadModel()).Times(AnyNumber());
    EXPECT_CALL(*raw_backend, infer(_, _))
        .WillOnce([&](const Batch&, std::vector<float>&) {
            if (!oom_triggered) {
                oom_triggered = true;
                oom_seen.set_value();
                throw GpuMemoryPressureException("simulated OOM");
            }
        })
        .WillRepeatedly(Return());

    EXPECT_CALL(*raw_decoder, decode(_, _, _, _, _))
        .WillRepeatedly(Return(std::vector<std::vector<Detection>>{{}}));

    InferWorker worker(cfg,
                       std::unique_ptr<MockBackend>(raw_backend),
                       std::unique_ptr<MockDecoder>(raw_decoder),
                       publisher, nullptr, nullptr, nullptr, nullptr);
    worker.start();
    worker.enqueue(makeBatch());

    ASSERT_EQ(oom_future.wait_for(std::chrono::milliseconds(500)),
              std::future_status::ready);

    // Give worker time to process the re-queued batch
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const std::string out = Metrics::get().serialize();
    EXPECT_NE(out.find("gpu_oom_total{worker_id=\"iw_oom_m1\"} 1"), std::string::npos);

    // State must remain RUNNING after OOM
    EXPECT_EQ(worker.state(), WorkerState::RUNNING);

    worker.stop();
}

TEST(InferWorkerRecovery, EngineFaultTriggersReloadAndReturnsToRunning) {
    auto cfg = makeModelConfig("iw_fault_m2");
    auto* raw_backend = new MockBackend();
    auto* raw_decoder = new MockDecoder();
    MockPublisher publisher;

    std::promise<void> recovered;
    auto rec_future = recovered.get_future();
    bool reloaded = false;

    EXPECT_CALL(*raw_backend, loadModel(_))
        .WillRepeatedly([&](const ModelConfig&) {
            if (reloaded) recovered.set_value();
            reloaded = true;
        });
    EXPECT_CALL(*raw_backend, unloadModel()).Times(AnyNumber());
    EXPECT_CALL(*raw_backend, infer(_, _))
        .WillOnce(Throw(GpuFaultException(GpuFaultType::ENGINE_FAULT, "trt crash")))
        .WillRepeatedly(Return());

    EXPECT_CALL(*raw_decoder, decode(_, _, _, _, _))
        .WillRepeatedly(Return(std::vector<std::vector<Detection>>{{}}));

    InferWorker worker(cfg,
                       std::unique_ptr<MockBackend>(raw_backend),
                       std::unique_ptr<MockDecoder>(raw_decoder),
                       publisher, nullptr, nullptr, nullptr, nullptr);
    worker.start();
    worker.enqueue(makeBatch());

    // Wait for recovery reload to complete (up to 2s)
    ASSERT_EQ(rec_future.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(worker.state(), WorkerState::RUNNING);

    const std::string out = Metrics::get().serialize();
    EXPECT_NE(out.find("gpu_engine_fault_total{worker_id=\"iw_fault_m2\"} 1"),
              std::string::npos);

    worker.stop();
}

TEST(InferWorkerRecovery, AllLoadRetriesFailLeadsToStopped) {
    auto cfg = makeModelConfig("iw_dead_m3");
    auto* raw_backend = new MockBackend();
    auto* raw_decoder = new MockDecoder();
    MockPublisher publisher;

    std::promise<void> stopped_seen;
    auto stopped_future = stopped_seen.get_future();

    // First loadModel (start) succeeds; all recovery reloads fail
    int load_count = 0;
    EXPECT_CALL(*raw_backend, loadModel(_))
        .WillRepeatedly([&](const ModelConfig&) {
            if (load_count++ > 0)
                throw std::runtime_error("device lost");
        });
    EXPECT_CALL(*raw_backend, unloadModel()).Times(AnyNumber());
    EXPECT_CALL(*raw_backend, infer(_, _))
        .WillOnce(Throw(GpuFaultException(GpuFaultType::ENGINE_FAULT, "crash")));
    EXPECT_CALL(*raw_decoder, decode(_, _, _, _, _))
        .WillRepeatedly(Return(std::vector<std::vector<Detection>>{{}}));

    InferWorker worker(cfg,
                       std::unique_ptr<MockBackend>(raw_backend),
                       std::unique_ptr<MockDecoder>(raw_decoder),
                       publisher, nullptr, nullptr, nullptr, nullptr);
    worker.start();
    worker.enqueue(makeBatch());

    // Wait for STOPPED state (recovery exhausted, 3 retries with 1s/2s/4s backoff → ~7s max)
    // Use polling to avoid waiting the full backoff in tests
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (worker.state() == WorkerState::STOPPED) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_EQ(worker.state(), WorkerState::STOPPED);

    worker.stop(); // safe to call even when already stopped
}
