#include <gtest/gtest.h>
#include "common/Config.h"
#include <stdexcept>
#include <fstream>
#include <cstdio>

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
    EXPECT_EQ(cfg.server.management_port,     9090);
    EXPECT_EQ(cfg.server.ffmpeg_log_level,    "fatal");
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

    // kafka (via legacy fallback into publishers.kafka)
    EXPECT_EQ(cfg.publishers.kafka.brokers, "localhost:9092");
    EXPECT_EQ(cfg.publishers.kafka.topic,   "test-results");
    EXPECT_EQ(cfg.publishers.kafka.batch_size, 50);

    // frame archive
    EXPECT_TRUE(cfg.frame_archive.enabled);
    EXPECT_FALSE(cfg.frame_archive.allow_gpu_frames);
    EXPECT_EQ(cfg.frame_archive.worker_count, 3);
    EXPECT_EQ(cfg.frame_archive.local_dir, "/tmp/infer-frames");
    EXPECT_EQ(cfg.frame_archive.save_interval, 2);
    EXPECT_EQ(cfg.frame_archive.jpeg_quality, 85);
    EXPECT_EQ(cfg.frame_archive.queue_capacity, 128);
    EXPECT_TRUE(cfg.frame_archive.minio.enabled);
    EXPECT_EQ(cfg.frame_archive.minio.endpoint, "minio.local:9000");
    EXPECT_EQ(cfg.frame_archive.minio.bucket, "infer-frames");
    EXPECT_EQ(cfg.frame_archive.minio.access_key, "test_access");
    EXPECT_EQ(cfg.frame_archive.minio.secret_key, "test_secret");
    EXPECT_EQ(cfg.frame_archive.minio.region, "us-east-1");
    EXPECT_FALSE(cfg.frame_archive.minio.use_ssl);
    EXPECT_EQ(cfg.frame_archive.minio.connect_timeout_ms, 1200);
    EXPECT_EQ(cfg.frame_archive.minio.request_timeout_ms, 2800);
    EXPECT_EQ(cfg.frame_archive.minio.max_retries, 3);
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

TEST(LoadConfig, ParsePublishersBlock) {
    AppConfig cfg = loadConfig("data/test_config_publishers.yaml");
    EXPECT_TRUE(cfg.publishers.kafka.enabled);
    EXPECT_EQ(cfg.publishers.kafka.brokers, "broker:9093");
    EXPECT_EQ(cfg.publishers.kafka.topic, "pub-results");
    EXPECT_EQ(cfg.publishers.kafka.batch_size, 200);
    EXPECT_EQ(cfg.publishers.kafka.linger_ms, 15);
    EXPECT_EQ(cfg.publishers.kafka.compression, "gzip");
    EXPECT_EQ(cfg.publishers.kafka.queue_capacity, 5000);
    EXPECT_EQ(cfg.publishers.kafka.heartbeat_topic, "pub-heartbeat");
    EXPECT_EQ(cfg.publishers.kafka.heartbeat_interval_ms, 3000);
    EXPECT_EQ(cfg.publishers.kafka.control_topic, "pub-control");
    EXPECT_TRUE(cfg.publishers.grpc.enabled);
    EXPECT_EQ(cfg.publishers.grpc.port, 50052);
    EXPECT_EQ(cfg.publishers.grpc.max_connections, 50);
    EXPECT_TRUE(cfg.publishers.redis.enabled);
    EXPECT_EQ(cfg.publishers.redis.host, "redis-host");
    EXPECT_EQ(cfg.publishers.redis.port, 6380);
    EXPECT_EQ(cfg.publishers.redis.stream_prefix, "infer");
    EXPECT_EQ(cfg.publishers.redis.max_len, 500);
    EXPECT_EQ(cfg.publishers.redis.queue_capacity, 2000);
}

TEST(LoadConfig, LegacyKafkaKeyFallback) {
    AppConfig cfg = loadConfig("data/test_config.yaml");
    EXPECT_TRUE(cfg.publishers.kafka.enabled);
    EXPECT_EQ(cfg.publishers.kafka.brokers, "localhost:9092");
    EXPECT_EQ(cfg.publishers.kafka.topic, "test-results");
}

TEST(LoadConfig, NoPublisherEnabledThrows) {
    const std::string path = "data/test_no_publisher.yaml";
    {
        std::ofstream out(path);
        out << "sources:\n  - id: cam_1\n    url: rtsp://localhost/test\n";
        out << "models:\n  - id: m1\n    version: yolov8\n    backend: tensorrt\n    input_size: [640, 640]\n";
        out << "pipelines:\n  - id: p1\n    nodes:\n      - id: cam_1\n        type: source.rtsp\n";
        out << "      - id: sink_1\n        type: sink.kafka\n    edges:\n      - from: cam_1\n        to: sink_1\n";
        out << "tasks:\n  - id: t1\n    source_id: cam_1\n    pipeline_id: p1\n";
        out << "publishers:\n  kafka:\n    enabled: false\n  grpc:\n    enabled: false\n  redis:\n    enabled: false\n";
    }
    EXPECT_THROW(loadConfig(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(LoadConfig, GrpcConfigDefaults) {
    AppConfig cfg = loadConfig("data/test_config.yaml");
    EXPECT_FALSE(cfg.publishers.grpc.enabled);
    EXPECT_EQ(cfg.publishers.grpc.port, 50051);
    EXPECT_EQ(cfg.publishers.grpc.max_connections, 100);
}

TEST(LoadConfig, RedisConfigDefaults) {
    AppConfig cfg = loadConfig("data/test_config.yaml");
    EXPECT_FALSE(cfg.publishers.redis.enabled);
    EXPECT_EQ(cfg.publishers.redis.host, "localhost");
    EXPECT_EQ(cfg.publishers.redis.port, 6379);
    EXPECT_EQ(cfg.publishers.redis.stream_prefix, "inference");
    EXPECT_EQ(cfg.publishers.redis.max_len, 1000);
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
