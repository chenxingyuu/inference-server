#include <gtest/gtest.h>

#include "pipeline/StageFactory.h"

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
    return StageFactory::Context{cfg, source, publisher, nullptr, 5, false};
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

} // namespace infer
