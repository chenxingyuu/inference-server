#include <gtest/gtest.h>

#include "pipeline/StageFactory.h"
#include "pipeline/stages/InferEngineWorkerStage.h"

namespace infer {
namespace {

class DummyPublisher final : public IPublisher {
public:
    void publish(InferResult) override {}
    void flush() override {}
};

StageFactory::Context makeContext() {
    static AppConfig cfg;
    static PipelineSourceConfig source;
    source.id = "cam_01";
    source.url = "rtsp://localhost/test";
    source.reconnect_delay_ms = 1000;
    source.max_reconnect_delay_ms = 5000;
    source.degraded_threshold = 5;
    source.max_reconnect_attempts = 5;
    static DummyPublisher publisher;
    return StageFactory::Context{cfg, source, publisher, nullptr, 5, SamplingMode::FrameCount, false};
}

StageFactory::Context makeInferEngineContext() {
    static AppConfig cfg;
    static PipelineSourceConfig source;
    static DummyPublisher publisher;
    static bool inited = false;
    if (!inited) {
        inited = true;
        source.id = "cam_01";
        source.url = "rtsp://localhost/test";
        source.reconnect_delay_ms = 1000;
        source.max_reconnect_delay_ms = 5000;
        source.degraded_threshold = 5;
        source.max_reconnect_attempts = 5;

        ModelConfig m;
        m.id = "model_multi";
        m.version = YOLOVersion::v8;
        m.backend = DeviceType::CPU;
        m.onnx_path = "models/placeholder.onnx";
        m.batch_size = 4;
        m.instance_count = 2;
        m.device_ids = {1, 2};
        m.num_classes = 80;
        m.input_shape.batch = 4;
        m.input_shape.channels = 3;
        m.input_shape.height = 640;
        m.input_shape.width = 640;
        cfg.models.push_back(m);
    }
    return StageFactory::Context{cfg, source, publisher, nullptr, 5, SamplingMode::FrameCount, false};
}

} // namespace

TEST(StageFactory, CreatesSinkStreamStage) {
    StageConfig cfg;
    cfg.id = "stream_sink_1";
    cfg.type = "sink.stream";
    cfg.with["output_url"] = "rtsp://localhost:8554/live/test";
    cfg.with["protocol"] = "rtsp";

    EXPECT_NO_THROW({
        auto stage = StageFactory::create(cfg, makeContext());
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->id(), "stream_sink_1");
    });
}

TEST(StageFactory, RejectsUnsupportedSinkStreamProtocol) {
    StageConfig cfg;
    cfg.id = "stream_sink_1";
    cfg.type = "sink.stream";
    cfg.with["output_url"] = "rtsp://localhost:8554/live/test";
    cfg.with["protocol"] = "srt";

    EXPECT_THROW((void)StageFactory::create(cfg, makeContext()), std::runtime_error);
}

TEST(StageFactory, RequiresSinkStreamOutputUrl) {
    StageConfig cfg;
    cfg.id = "stream_sink_1";
    cfg.type = "sink.stream";
    cfg.with["protocol"] = "rtsp";

    EXPECT_THROW((void)StageFactory::create(cfg, makeContext()), std::runtime_error);
}

TEST(StageFactory, CreatesSinkFfplayStage) {
    StageConfig cfg;
    cfg.id = "ffplay_sink_1";
    cfg.type = "sink.ffplay";
    cfg.with["fps"] = "12";

    EXPECT_NO_THROW({
        auto stage = StageFactory::create(cfg, makeContext());
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->id(), "ffplay_sink_1");
    });
}

// ─── source.file ─────────────────────────────────────────────────────────────

TEST(StageFactory, CreatesSourceFileStage) {
    StageConfig cfg;
    cfg.id   = "vid_001";
    cfg.type = "source.file";

    auto ctx = makeContext();
    EXPECT_NO_THROW({
        auto stage = StageFactory::create(cfg, ctx);
        ASSERT_NE(stage, nullptr);
        EXPECT_EQ(stage->id(), "vid_001");
        EXPECT_TRUE(stage->isSource());
    });
}

TEST(StageFactory, CreatesSourceFileStageWithLoop) {
    StageConfig cfg;
    cfg.id   = "vid_loop";
    cfg.type = "source.file";
    cfg.with["loop"] = "true";

    auto ctx = makeContext();
    EXPECT_NO_THROW({
        auto stage = StageFactory::create(cfg, ctx);
        ASSERT_NE(stage, nullptr);
        EXPECT_TRUE(stage->isSource());
    });
}

#if defined(BUILD_ONNX_BACKEND)
TEST(StageFactory, InferEngineCreatesInferEngineWorkerStage) {
    StageConfig cfg;
    cfg.id = "infer_1";
    cfg.type = "infer.engine";
    cfg.with["model_id"] = "model_multi";

    auto ctx = makeInferEngineContext();
    auto stage = StageFactory::create(cfg, ctx);
    ASSERT_NE(stage, nullptr);
    auto* worker_stage = dynamic_cast<InferEngineWorkerStage*>(stage.get());
    ASSERT_NE(worker_stage, nullptr);
    EXPECT_EQ(worker_stage->workerInstanceCount(), 2);
}
#endif

} // namespace infer
