#include <gtest/gtest.h>
#include "common/Config.h"
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <filesystem>

using namespace infer;

// ── parseYOLOVersion ──────────────────────────────────────────────────────

TEST(ParseYOLOVersion, KnownStrings) {
    EXPECT_EQ(parseYOLOVersion("yolov5"), YOLOVersion::v5);
    EXPECT_EQ(parseYOLOVersion("v5"),     YOLOVersion::v5);
    EXPECT_EQ(parseYOLOVersion("yolov8"), YOLOVersion::v8);
    EXPECT_EQ(parseYOLOVersion("v8"),     YOLOVersion::v8);
    EXPECT_EQ(parseYOLOVersion("yolo11"), YOLOVersion::v11);
    EXPECT_EQ(parseYOLOVersion("v11"),    YOLOVersion::v11);
    EXPECT_EQ(parseYOLOVersion("yolo26"), YOLOVersion::v26);
    EXPECT_EQ(parseYOLOVersion("v26"),    YOLOVersion::v26);
}

TEST(ParseYOLOVersion, UnknownStringThrows) {
    EXPECT_THROW(parseYOLOVersion("yolov4"), std::runtime_error);
    EXPECT_THROW(parseYOLOVersion(""),       std::runtime_error);
    EXPECT_THROW(parseYOLOVersion("YOLOV8"), std::runtime_error);
}

// ── parseDeviceType ───────────────────────────────────────────────────────

TEST(ParseDeviceType, KnownStrings) {
    EXPECT_EQ(parseDeviceType("tensorrt"), DeviceType::CUDA);
    EXPECT_EQ(parseDeviceType("cuda"),     DeviceType::CUDA);
    EXPECT_EQ(parseDeviceType("trt"),      DeviceType::CUDA);
    EXPECT_EQ(parseDeviceType("ascend"),   DeviceType::Ascend);
    EXPECT_EQ(parseDeviceType("acl"),      DeviceType::Ascend);
    EXPECT_EQ(parseDeviceType("cpu"),      DeviceType::CPU);
}

TEST(ParseDeviceType, UnknownStringThrows) {
    EXPECT_THROW(parseDeviceType("gpu"),   std::runtime_error);
    EXPECT_THROW(parseDeviceType(""),      std::runtime_error);
    EXPECT_THROW(parseDeviceType("CUDA"),  std::runtime_error);
}

// ── parseTrackerType ──────────────────────────────────────────────────────

TEST(ParseTrackerType, KnownStrings) {
    EXPECT_EQ(parseTrackerType("none"), TrackerType::None);
    EXPECT_EQ(parseTrackerType("bytetrack"), TrackerType::ByteTrack);
    EXPECT_EQ(parseTrackerType("deepsort"), TrackerType::DeepSort);
}

TEST(ParseTrackerType, UnknownStringThrows) {
    EXPECT_THROW(parseTrackerType("byte_track"), std::runtime_error);
    EXPECT_THROW(parseTrackerType(""), std::runtime_error);
}

// ── AppConfig::findModel / findStream ─────────────────────────────────────

class AppConfigFindTest : public ::testing::Test {
protected:
    void SetUp() override {
        ModelConfig m1; m1.id = "det_01";
        ModelConfig m2; m2.id = "cls_01";
        cfg.models = {m1, m2};
        PipelineSourceConfig s1; s1.id = "cam_01";
        PipelineSourceConfig s2; s2.id = "cam_02";
        cfg.sources = {s1, s2};
    }
    AppConfig cfg;
};

TEST_F(AppConfigFindTest, FindExistingModel) {
    const ModelConfig* m = cfg.findModel("det_01");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->id, "det_01");
}

TEST_F(AppConfigFindTest, FindMissingModelReturnsNull) {
    EXPECT_EQ(cfg.findModel("nonexistent"), nullptr);
}

TEST_F(AppConfigFindTest, FindExistingStream) {
    const PipelineSourceConfig* s = cfg.findSource("cam_02");
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->id, "cam_02");
}

TEST_F(AppConfigFindTest, FindMissingStreamReturnsNull) {
    EXPECT_EQ(cfg.findSource("cam_99"), nullptr);
}

TEST_F(AppConfigFindTest, FindTask) {
    TaskConfig t;
    t.id = "t1";
    t.source_id = "cam_01";
    t.pipeline_id = "pipe_01";
    cfg.tasks.push_back(t);
    const TaskConfig* found = cfg.findTask("t1");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->source_id, "cam_01");
    EXPECT_EQ(cfg.findTask("none"), nullptr);
}

// ── loadConfig from YAML ──────────────────────────────────────────────────

TEST(LoadConfig, ParsesFullYaml) {
    AppConfig cfg = loadConfig("data/test_config.yaml");

    // server
    EXPECT_EQ(cfg.server.stream_pool_threads, 4);
    EXPECT_EQ(cfg.server.max_streams,         10);
    EXPECT_EQ(cfg.server.socket_path,         "/var/run/infer.sock");
    EXPECT_EQ(cfg.server.ffmpeg_log_level,    "fatal");
    EXPECT_EQ(cfg.server.ffmpeg_decode_threads, 4);
    EXPECT_EQ(cfg.server.log_level,           "debug");

    // models
    ASSERT_EQ(cfg.models.size(), 2u);
    const ModelConfig& det = cfg.models[0];
    EXPECT_EQ(det.id,          "yolo_det");
    EXPECT_EQ(det.version,     YOLOVersion::v5);
    EXPECT_EQ(det.backend,     DeviceType::CUDA);
    EXPECT_EQ(det.batch_size,  8);
    EXPECT_FLOAT_EQ(det.conf_thresh, 0.5f);
    EXPECT_FLOAT_EQ(det.nms_thresh,  0.4f);
    EXPECT_EQ(det.num_classes, 3);
    EXPECT_EQ(det.input_shape.height, 640);
    EXPECT_EQ(det.input_shape.width,  640);
    ASSERT_EQ(det.class_names.size(), 3u);
    EXPECT_EQ(det.class_names[0], "car");
    EXPECT_EQ(det.class_names[1], "truck");
    EXPECT_EQ(det.class_names[2], "bus");

    // cascade
    ASSERT_EQ(det.cascade.size(), 1u);
    EXPECT_EQ(det.cascade[0].model_id,      "classifier_01");
    EXPECT_EQ(det.cascade[0].attribute_key, "vehicle_type");
    EXPECT_FLOAT_EQ(det.cascade[0].crop_expand, 0.1f);
    ASSERT_EQ(det.cascade[0].trigger_classes.size(), 2u);
    EXPECT_EQ(det.cascade[0].trigger_classes[0], 0);
    EXPECT_EQ(det.cascade[0].trigger_classes[1], 1);

    // classifier model
    const ModelConfig& cls = cfg.models[1];
    EXPECT_EQ(cls.id,         "classifier_01");
    EXPECT_EQ(cls.model_type, ModelType::Classifier);

    // sources (ingest: sample_fps / use_hwdec live on tasks)
    ASSERT_EQ(cfg.sources.size(), 2u);
    EXPECT_EQ(cfg.sources[0].id, "cam_01");
    EXPECT_EQ(cfg.sources[0].url, "rtsp://localhost/cam1");
    EXPECT_EQ(cfg.sources[0].reconnect_delay_ms, 1000);
    EXPECT_EQ(cfg.sources[1].id, "cam_02");
    EXPECT_EQ(cfg.sources[1].reconnect_delay_ms, 2000);

    // pipelines (templates, no source_id)
    ASSERT_EQ(cfg.pipelines.size(), 1u);
    EXPECT_EQ(cfg.pipelines[0].id, "pipe_01");
    ASSERT_EQ(cfg.pipelines[0].nodes.size(), 9u);
    ASSERT_EQ(cfg.pipelines[0].edges.size(), 9u);
    EXPECT_EQ(cfg.pipelines[0].nodes[4].type, "infer.engine");
    EXPECT_EQ(cfg.pipelines[0].nodes[4].with.at("model_id"), "yolo_det");

    // tasks
    ASSERT_EQ(cfg.tasks.size(), 1u);
    EXPECT_EQ(cfg.tasks[0].id, "task_pipe_01");
    EXPECT_EQ(cfg.tasks[0].source_id, "cam_01");
    EXPECT_EQ(cfg.tasks[0].pipeline_id, "pipe_01");
    EXPECT_EQ(cfg.tasks[0].sample_fps, 10);
    EXPECT_FALSE(cfg.tasks[0].use_hwdec);
    EXPECT_TRUE(cfg.tasks[0].use_ascend_dvpp);
    EXPECT_EQ(cfg.tasks[0].ascend_device_id, 2);
    const TaskConfig* t = cfg.findTask("task_pipe_01");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->pipeline_id, "pipe_01");
    EXPECT_TRUE(t->use_ascend_dvpp);
    EXPECT_EQ(t->ascend_device_id, 2);
    EXPECT_EQ(cfg.findTask("missing"), nullptr);

    // kafka publisher
    ASSERT_EQ(cfg.publishers.size(), 1u);
    EXPECT_EQ(cfg.publishers[0].id,   "kafka_main");
    EXPECT_EQ(cfg.publishers[0].type, "kafka");
    EXPECT_EQ(cfg.publishers[0].kafka.brokers,    "localhost:9092");
    EXPECT_EQ(cfg.publishers[0].kafka.topic,      "test-results");
    EXPECT_EQ(cfg.publishers[0].kafka.batch_size, 50);

    // frame archive
    EXPECT_TRUE(cfg.frame_archive.enabled);
    EXPECT_FALSE(cfg.frame_archive.allow_gpu_frames);
    EXPECT_EQ(cfg.frame_archive.worker_count, 3);
    EXPECT_EQ(cfg.frame_archive.local_dir, "/tmp/infer-frames");
    EXPECT_EQ(cfg.frame_archive.save_interval, 2);
    EXPECT_EQ(cfg.frame_archive.jpeg_quality, 85);
    EXPECT_EQ(cfg.frame_archive.queue_capacity, 128);
}

TEST(LoadConfig, LogLevelDefaultsToInfo) {
    const std::string path = "data/test_log_level_default.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    AppConfig cfg = loadConfig(path);
    EXPECT_EQ(cfg.server.log_level, "info");
    std::remove(path.c_str());
}

TEST(LoadConfig, MissingFileThrows) {
    EXPECT_THROW(loadConfig("nonexistent.yaml"), std::runtime_error);
}

TEST(LoadConfig, InvalidEdgePolicyThrows) {
    const std::string path = "data/test_invalid_tracker.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "        queue:\n";
        out << "          drop_policy: bad_policy\n";
        out << "tasks:\n";
        out << "  - id: task1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, MissingTasksKeyThrows) {
    const std::string path = "data/test_no_tasks_key.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, EmptyTasksThrows) {
    const std::string path = "data/test_empty_tasks.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks: []\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, DuplicateTaskIdThrows) {
    const std::string path = "data/test_dup_tasks.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: same_id\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
        out << "  - id: same_id\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, TaskUnknownSourceThrows) {
    const std::string path = "data/test_task_bad_source.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: no_such_cam\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, TaskUnknownPipelineThrows) {
    const std::string path = "data/test_task_bad_pipeline.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: no_such_pipe\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, DeprecatedSourceSampleFpsThrows) {
    const std::string path = "data/test_deprecated_source_sample_fps.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "    sample_fps: 5\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, TaskSampleFpsBelowOneThrows) {
    const std::string path = "data/test_task_bad_sample_fps.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
        out << "    sample_fps: 0\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, TaskAscendDeviceIdNegativeThrows) {
    const std::string path = "data/test_task_bad_ascend_device_id.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: ascend\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
        out << "    use_ascend_dvpp: true\n";
        out << "    ascend_device_id: -1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, InvalidPipelineGraphThrows) {
    const std::string path = "data/test_invalid_bytetrack_params.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: stage_2\n";
        out << "        type: decode.ffmpeg\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: stage_2\n";
        out << "      - from: stage_2\n";
        out << "        to: cam_1\n";
        out << "tasks:\n";
        out << "  - id: task1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, SinkStreamInvalidProtocolThrows) {
    const std::string path = "data/test_invalid_sink_stream_protocol.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_stream_1\n";
        out << "        type: sink.stream\n";
        out << "        with:\n";
        out << "          output_url: rtmp://localhost/live/test\n";
        out << "          protocol: srt\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_stream_1\n";
        out << "tasks:\n";
        out << "  - id: task1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, SourceIdPathTraversalStyleRejected) {
    const std::string path = "data/test_invalid_source_id.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: ../cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: source_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: source_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: ../cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, SinkFfplayInvalidFpsRejected) {
    const std::string path = "data/test_invalid_sink_ffplay_fps.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: source_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.ffplay\n";
        out << "        with:\n";
        out << "          fps: .nan\n";
        out << "    edges:\n";
        out << "      - from: source_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── Publishers config ─────────────────────────────────────────────────────

namespace {
const PublisherConfig* findPub(const std::vector<PublisherConfig>& pubs, const std::string& id) {
    for (const auto& p : pubs) if (p.id == id) return &p;
    return nullptr;
}
} // namespace

TEST(LoadConfig, ParsePublishersBlock) {
    AppConfig cfg = loadConfig("data/test_config_publishers.yaml");
    ASSERT_EQ(cfg.publishers.size(), 3u);

    const auto* kafka = findPub(cfg.publishers, "kafka_main");
    ASSERT_NE(kafka, nullptr);
    EXPECT_EQ(kafka->type, "kafka");
    EXPECT_EQ(kafka->kafka.brokers,        "broker:9093");
    EXPECT_EQ(kafka->kafka.topic,          "pub-results");
    EXPECT_EQ(kafka->kafka.batch_size,     200);
    EXPECT_EQ(kafka->kafka.linger_ms,      15);
    EXPECT_EQ(kafka->kafka.compression,    "gzip");
    EXPECT_EQ(kafka->kafka.queue_capacity, 5000);

    const auto* grpc = findPub(cfg.publishers, "grpc_rt");
    ASSERT_NE(grpc, nullptr);
    EXPECT_EQ(grpc->type, "grpc");
    EXPECT_EQ(grpc->grpc.port,            50052);
    EXPECT_EQ(grpc->grpc.max_connections, 50);

    const auto* redis = findPub(cfg.publishers, "redis_stream");
    ASSERT_NE(redis, nullptr);
    EXPECT_EQ(redis->type, "redis");
    EXPECT_EQ(redis->redis.host,           "redis-host");
    EXPECT_EQ(redis->redis.port,           6380);
    EXPECT_EQ(redis->redis.stream_prefix,  "infer");
    EXPECT_EQ(redis->redis.max_len,        500);
    EXPECT_EQ(redis->redis.queue_capacity, 2000);
}

TEST(LoadConfig, NoPublishersThrows) {
    const std::string path = "data/test_no_publisher.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n  - id: cam_1\n    url: rtsp://localhost/test\n";
        out << "models:\n  - id: m1\n    version: yolov8\n    backend: tensorrt\n    input_size: [640, 640]\n";
        out << "pipelines:\n  - id: p1\n    nodes:\n      - id: cam_1\n        type: source.rtsp\n";
        out << "      - id: sink_1\n        type: sink.kafka\n    edges:\n      - from: cam_1\n        to: sink_1\n";
        out << "tasks:\n  - id: t1\n    source_id: cam_1\n    pipeline_id: p1\n";
        out << "publishers: []\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, PublisherDuplicateIdThrows) {
    const std::string path = "data/test_dup_publisher.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n  - id: cam_1\n    url: rtsp://localhost/test\n";
        out << "models:\n  - id: m1\n    version: yolov8\n    backend: tensorrt\n    input_size: [640, 640]\n";
        out << "pipelines:\n  - id: p1\n    nodes:\n      - id: cam_1\n        type: source.rtsp\n";
        out << "      - id: sink_1\n        type: sink.kafka\n    edges:\n      - from: cam_1\n        to: sink_1\n";
        out << "tasks:\n  - id: t1\n    source_id: cam_1\n    pipeline_id: p1\n";
        out << "publishers:\n  - id: k1\n    type: kafka\n  - id: k1\n    type: kafka\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, PublisherUnknownTypeThrows) {
    const std::string path = "data/test_bad_pub_type.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n  - id: cam_1\n    url: rtsp://localhost/test\n";
        out << "models:\n  - id: m1\n    version: yolov8\n    backend: tensorrt\n    input_size: [640, 640]\n";
        out << "pipelines:\n  - id: p1\n    nodes:\n      - id: cam_1\n        type: source.rtsp\n";
        out << "      - id: sink_1\n        type: sink.kafka\n    edges:\n      - from: cam_1\n        to: sink_1\n";
        out << "tasks:\n  - id: t1\n    source_id: cam_1\n    pipeline_id: p1\n";
        out << "publishers:\n  - id: p1\n    type: unknown_type\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, FrameArchiveWorkerCountDefaultsToOne) {
    const std::string path = "data/test_frame_archive_worker_default.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
        out << "frame_archive:\n";
        out << "  enabled: true\n";
    }
    AppConfig cfg = loadConfig(path);
    EXPECT_EQ(cfg.frame_archive.worker_count, 1);
    std::remove(path.c_str());
}

TEST(LoadConfig, FrameArchiveWorkerCountMustBePositive) {
    const std::string path = "data/test_frame_archive_worker_invalid.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n";
        out << "  - id: cam_1\n";
        out << "    url: rtsp://localhost/test\n";
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "pipelines:\n";
        out << "  - id: p1\n";
        out << "    nodes:\n";
        out << "      - id: cam_1\n";
        out << "        type: source.rtsp\n";
        out << "      - id: sink_1\n";
        out << "        type: sink.kafka\n";
        out << "    edges:\n";
        out << "      - from: cam_1\n";
        out << "        to: sink_1\n";
        out << "tasks:\n";
        out << "  - id: t1\n";
        out << "    source_id: cam_1\n";
        out << "    pipeline_id: p1\n";
        out << "frame_archive:\n";
        out << "  enabled: true\n";
        out << "  worker_count: 0\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

// ── device_ids validation ─────────────────────────────────────────────────

static void writeMinimalYamlHeader(std::ostream& out) {
    out << "sources:\n";
    out << "  - id: cam_1\n";
    out << "    url: rtsp://localhost/test\n";
    out << "pipelines:\n";
    out << "  - id: p1\n";
    out << "    nodes:\n";
    out << "      - id: cam_1\n";
    out << "        type: source.rtsp\n";
    out << "      - id: sink_1\n";
    out << "        type: sink.kafka\n";
    out << "    edges:\n";
    out << "      - from: cam_1\n";
    out << "        to: sink_1\n";
    out << "tasks:\n";
    out << "  - id: t1\n";
    out << "    source_id: cam_1\n";
    out << "    pipeline_id: p1\n";
}

TEST(LoadConfig, DeviceIdsSizeMismatchInstanceCountThrows) {
    const std::string path = "data/test_device_ids_mismatch.yaml";
    {
        std::ofstream out(path);
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: ascend\n";
        out << "    input_size: [640, 640]\n";
        out << "    instance_count: 3\n";
        out << "    device_ids: [0, 1]\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, DeviceIdsSizeMatchingInstanceCountOk) {
    const std::string path = "data/test_device_ids_match.yaml";
    {
        std::ofstream out(path);
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: ascend\n";
        out << "    input_size: [640, 640]\n";
        out << "    instance_count: 2\n";
        out << "    device_ids: [0, 1]\n";
    }
    EXPECT_NO_THROW(loadConfig(path));
    std::remove(path.c_str());
}

TEST(LoadConfig, EmptyDeviceIdsWithMultipleInstancesOk) {
    const std::string path = "data/test_device_ids_empty.yaml";
    {
        std::ofstream out(path);
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "    instance_count: 4\n";
    }
    EXPECT_NO_THROW(loadConfig(path));
    std::remove(path.c_str());
}

// ── model_repository (filesystem registry) ────────────────────────────────

namespace fs = std::filesystem;

TEST(ModelRepository, ScanResolvesSingleOnnx) {
    const fs::path base =
        fs::temp_directory_path() / ("infer_repo_scan_" + std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(base / "repo_m" / "1");
    {
        std::ofstream m(base / "repo_m" / "1" / "model.onnx");
        m.put('x');
    }
    {
        std::ofstream cfg(base / "repo_m" / "config.yaml");
        cfg << "backend: cpu\n";
        cfg << "version: yolov8\n";
        cfg << "input_size: [640, 640]\n";
    }
    std::vector<ModelConfig> out = scanModelRepository(base.generic_string());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].id, "repo_m");
    EXPECT_EQ(out[0].backend, DeviceType::CPU);
    ASSERT_FALSE(out[0].onnx_path.empty());
    EXPECT_NE(out[0].onnx_path.find(std::string("/1/")), std::string::npos);
    fs::remove_all(base);
}

TEST(ModelRepository, ActiveVersionSelectsOlder) {
    const fs::path base =
        fs::temp_directory_path() / ("infer_repo_ver_" + std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(base / "ver_m" / "1");
    fs::create_directories(base / "ver_m" / "2");
    {
        std::ofstream a(base / "ver_m" / "1" / "v1.onnx");
        a.put('a');
    }
    {
        std::ofstream b(base / "ver_m" / "2" / "v2.onnx");
        b.put('b');
    }
    {
        std::ofstream cfg(base / "ver_m" / "config.yaml");
        cfg << "backend: cpu\n";
        cfg << "version: yolov8\n";
        cfg << "input_size: [640, 640]\n";
        cfg << "active_version: 1\n";
        cfg << "weight_file: v1.onnx\n";
    }
    std::vector<ModelConfig> out = scanModelRepository(base.generic_string());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NE(out[0].onnx_path.find("/1/"), std::string::npos);
    EXPECT_NE(out[0].onnx_path.find("v1.onnx"), std::string::npos);
    fs::remove_all(base);
}

TEST(ModelRepository, LoadConfigMergesRepositoryModels) {
    const fs::path base =
        fs::temp_directory_path() / ("infer_repo_merge_" + std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(base / "repo_only" / "1");
    {
        std::ofstream m(base / "repo_only" / "1" / "w.onnx");
        m.put('z');
    }
    {
        std::ofstream cfg(base / "repo_only" / "config.yaml");
        cfg << "backend: cpu\n";
        cfg << "version: yolov8\n";
        cfg << "input_size: [640, 640]\n";
    }
    const std::string path = "data/test_model_repo_merge.yaml";
    {
        std::ofstream out(path);
        out << "server:\n";
        out << "  model_repository: \"" << base.generic_string() << "\"\n";
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
    }
    AppConfig cfg = loadConfig(path);
    ASSERT_EQ(cfg.models.size(), 2u);
    ASSERT_NE(cfg.findModel("m1"), nullptr);
    const ModelConfig* rm = cfg.findModel("repo_only");
    ASSERT_NE(rm, nullptr);
    EXPECT_EQ(rm->backend, DeviceType::CPU);
    EXPECT_FALSE(rm->onnx_path.empty());
    std::remove(path.c_str());
    fs::remove_all(base);
}

TEST(ModelRepository, DuplicateIdBetweenRootAndRepoThrows) {
    const fs::path base =
        fs::temp_directory_path() / ("infer_repo_dup_" + std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(base / "m1" / "1");
    {
        std::ofstream m(base / "m1" / "1" / "x.onnx");
        m.put('q');
    }
    {
        std::ofstream cfg(base / "m1" / "config.yaml");
        cfg << "backend: cpu\n";
        cfg << "version: yolov8\n";
        cfg << "input_size: [640, 640]\n";
    }
    const std::string path = "data/test_model_repo_dup.yaml";
    {
        std::ofstream out(path);
        out << "server:\n";
        out << "  model_repository: \"" << base.generic_string() << "\"\n";
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: m1\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
    fs::remove_all(base);
}

TEST(LoadConfig, CascadeUnknownSecondaryThrows) {
    const std::string path = "data/test_cascade_unknown.yaml";
    {
        std::ofstream out(path);
        writeMinimalYamlHeader(out);
        out << "models:\n";
        out << "  - id: primary_m\n";
        out << "    version: yolov8\n";
        out << "    backend: tensorrt\n";
        out << "    input_size: [640, 640]\n";
        out << "    cascade:\n";
        out << "      - model_id: does_not_exist\n";
        out << "        crop_expand: 0.0\n";
        out << "        attribute_key: \"x\"\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}
