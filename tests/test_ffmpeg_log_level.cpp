#include <gtest/gtest.h>
#include "common/Config.h"
#include <fstream>
#include <cstdio>

using namespace infer;

// ─── ServerConfig defaults ────────────────────────────────────────────────────

TEST(FfmpegLogLevel, ServerConfigDefault) {
    ServerConfig cfg;
    EXPECT_EQ(cfg.ffmpeg_log_level, "warning");
}

// ─── YAML parsing ─────────────────────────────────────────────────────────────

static AppConfig loadFromString(const std::string& yaml) {
    const std::string path = "data/test_ffmpeg_log_level_tmp.yaml";
    {
        std::ofstream out(path);
        out << yaml;
    }
    auto cfg = loadConfig(path);
    std::remove(path.c_str());
    return cfg;
}

// Minimal valid YAML: one source, one pipeline with source+sink nodes, one task.
static const char* kMinimalYaml = R"(
sources:
  - id: cam_01
    url: rtsp://localhost/test
models:
  - id: det_01
    version: yolov8
    backend: cpu
    input_size: [640, 640]
pipelines:
  - id: pipe_01
    nodes:
      - id: cam_01
        type: source.rtsp
      - id: sink_01
        type: sink.kafka
    edges:
      - from: cam_01
        to: sink_01
tasks:
  - id: task_01
    source_id: cam_01
    pipeline_id: pipe_01
)";

TEST(FfmpegLogLevel, YamlParsesExplicitLevel) {
    std::string yaml = std::string("server:\n  ffmpeg_log_level: fatal\n") + kMinimalYaml;
    AppConfig cfg = loadFromString(yaml);
    EXPECT_EQ(cfg.server.ffmpeg_log_level, "fatal");
}

TEST(FfmpegLogLevel, YamlMissingFieldDefaultsToWarning) {
    std::string yaml = std::string("server:\n  management_port: 8080\n") + kMinimalYaml;
    AppConfig cfg = loadFromString(yaml);
    EXPECT_EQ(cfg.server.ffmpeg_log_level, "warning");
}
