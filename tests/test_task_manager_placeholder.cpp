#include <gtest/gtest.h>

#include "archive/FrameArchiver.h"
#include "pipeline/TaskManager.h"
#include "publisher/IPublisher.h"

#include <cstdio>

namespace infer {
namespace {

class NoopPublisher final : public IPublisher {
public:
    void publish(InferResult) override {}
    void flush() override {}
};

AppConfig makeBaseConfig(std::string output_url) {
    AppConfig cfg;

    PipelineSourceConfig src1;
    src1.id = "cam_01";
    src1.url = "rtsp://example.com/cam_01";
    PipelineSourceConfig src2;
    src2.id = "cam_02";
    src2.url = "rtsp://example.com/cam_02";
    cfg.sources = {src1, src2};

    PipelineConfig pipeline;
    pipeline.id = "pipe_stream";
    StageConfig source_stage;
    source_stage.id = "src";
    source_stage.type = "source.rtsp";
    StageConfig sink_stage;
    sink_stage.id = "sink";
    sink_stage.type = "sink.stream";
    sink_stage.with["output_url"] = std::move(output_url);
    sink_stage.with["protocol"] = "rtsp";
    pipeline.nodes = {source_stage, sink_stage};
    pipeline.edges = {EdgeConfig{"src", "sink"}};
    cfg.pipelines = {pipeline};

    TaskConfig t1;
    t1.id = "task_cam_01";
    t1.source_id = "cam_01";
    t1.pipeline_id = "pipe_stream";
    TaskConfig t2;
    t2.id = "task_cam_02";
    t2.source_id = "cam_02";
    t2.pipeline_id = "pipe_stream";
    cfg.tasks = {t1, t2};

    return cfg;
}

std::string makeTempConfigPath(const char* suffix) {
    return std::string("/tmp/infer_task_placeholder_") + suffix + "_" + std::to_string(::getpid()) + ".yaml";
}

TEST(TaskManagerPlaceholder, SinkStreamOutputUrlSupportsTaskAndSourcePlaceholders) {
    AppConfig cfg = makeBaseConfig("rtsp://push/live/{source_id}/{task_id}");
    NoopPublisher publisher;
    auto archiver = std::make_shared<FrameArchiver>(FrameArchiveConfig{});
    const std::string config_path = makeTempConfigPath("ok");

    TaskManager manager(std::move(cfg), config_path, publisher, archiver);
    EXPECT_NO_THROW(manager.loadAll());
    std::remove((config_path + ".state.json").c_str());
}

TEST(TaskManagerPlaceholder, SinkStreamOutputUrlUnknownPlaceholderThrows) {
    AppConfig cfg = makeBaseConfig("rtsp://push/live/{camera_id}");
    NoopPublisher publisher;
    auto archiver = std::make_shared<FrameArchiver>(FrameArchiveConfig{});
    const std::string config_path = makeTempConfigPath("bad");

    TaskManager manager(std::move(cfg), config_path, publisher, archiver);
    EXPECT_THROW(manager.loadAll(), std::runtime_error);
    std::remove((config_path + ".state.json").c_str());
}

TEST(TaskManagerPlaceholder, SinkStreamOutputUrlUnterminatedPlaceholderThrows) {
    AppConfig cfg = makeBaseConfig("rtsp://push/live/{task_id");
    NoopPublisher publisher;
    auto archiver = std::make_shared<FrameArchiver>(FrameArchiveConfig{});
    const std::string config_path = makeTempConfigPath("unterminated");

    TaskManager manager(std::move(cfg), config_path, publisher, archiver);
    EXPECT_THROW(manager.loadAll(), std::runtime_error);
    std::remove((config_path + ".state.json").c_str());
}

TEST(TaskManagerPlaceholder, SinkStreamOutputUrlWithoutPlaceholdersStillWorks) {
    AppConfig cfg = makeBaseConfig("rtsp://push/live/fixed");
    NoopPublisher publisher;
    auto archiver = std::make_shared<FrameArchiver>(FrameArchiveConfig{});
    const std::string config_path = makeTempConfigPath("compat");

    TaskManager manager(std::move(cfg), config_path, publisher, archiver);
    EXPECT_NO_THROW(manager.loadAll());
    std::remove((config_path + ".state.json").c_str());
}

} // namespace
} // namespace infer
