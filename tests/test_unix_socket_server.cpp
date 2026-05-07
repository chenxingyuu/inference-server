#include "server/UnixSocketServer.h"
#include "pipeline/ITaskManager.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <string>
#include <thread>
#include <chrono>
#include <cstdio>

using json = nlohmann::json;
using namespace infer;

// ── Fake TaskManager ──────────────────────────────────────────────────────────

struct FakeTaskManager : ITaskManager {
    std::vector<std::pair<std::string, State>> tasks;
    std::vector<SourceInfo>   sources;
    std::vector<PipelineInfo> pipelines;
    std::vector<ModelInfo>    models;
    bool start_result{true};
    bool stop_result{true};

    // Mutation result stubs — flip to false to simulate failure.
    bool add_source_result{true};
    bool remove_source_result{true};
    bool add_pipeline_result{true};
    bool remove_pipeline_result{true};
    bool add_task_result{true};
    bool remove_task_result{true};
    bool load_model_result{true};
    bool unload_model_result{true};

    std::vector<std::pair<std::string, State>> listTasks()    const override { return tasks; }
    std::vector<SourceInfo>                    listSources()  const override { return sources; }
    std::vector<PipelineInfo>                  listPipelines()const override { return pipelines; }
    std::vector<ModelInfo>                     listModels()   const override { return models; }
    bool start(const std::string&) override { return start_result; }
    bool stop(const std::string&) override { return stop_result; }

    bool addSource(const PipelineSourceConfig&) override { return add_source_result; }
    bool removeSource(const std::string&)       override { return remove_source_result; }
    bool addPipeline(const PipelineConfig&)     override { return add_pipeline_result; }
    bool removePipeline(const std::string&)     override { return remove_pipeline_result; }
    bool addTask(const TaskConfig&)             override { return add_task_result; }
    bool removeTask(const std::string&)         override { return remove_task_result; }
    bool loadModel(const ModelConfig&)          override { return load_model_result; }
    bool unloadModel(const std::string&)        override { return unload_model_result; }
};

// ── Test fixture ──────────────────────────────────────────────────────────────

class UnixSocketServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = "/tmp/test_infer_" + std::to_string(::getpid()) + ".sock";
        server_ = std::make_unique<UnixSocketServer>(path_, fake_tm_);
        server_->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    void TearDown() override {
        server_->stop();
        ::unlink(path_.c_str());
    }

    // Send one JSON line, return parsed response.
    json roundtrip(const json& req) {
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        ::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
        ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));

        std::string msg = req.dump() + "\n";
        ::send(fd, msg.data(), msg.size(), 0);

        std::string buf(4096, '\0');
        ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
        ::close(fd);

        buf.resize(n > 0 ? static_cast<size_t>(n) : 0);
        return json::parse(buf);
    }

    std::string path_;
    FakeTaskManager fake_tm_;
    std::unique_ptr<UnixSocketServer> server_;
};

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, HealthReturnsOk) {
    auto resp = roundtrip({{"cmd", "health"}});
    EXPECT_EQ(resp["status"], "ok");
}

TEST_F(UnixSocketServerTest, ListTasksEmpty) {
    auto resp = roundtrip({{"cmd", "list_tasks"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_TRUE(resp["data"].is_array());
    EXPECT_TRUE(resp["data"].empty());
}

TEST_F(UnixSocketServerTest, ListTasksWithItems) {
    fake_tm_.tasks = {{"cam0", ITaskManager::State::Running},
                      {"cam1", ITaskManager::State::Stopped}};
    auto resp = roundtrip({{"cmd", "list_tasks"}});
    ASSERT_EQ(resp["data"].size(), 2u);
    EXPECT_EQ(resp["data"][0]["id"],    "cam0");
    EXPECT_EQ(resp["data"][0]["state"], "running");
    EXPECT_EQ(resp["data"][1]["id"],    "cam1");
    EXPECT_EQ(resp["data"][1]["state"], "stopped");
}

TEST_F(UnixSocketServerTest, StartTaskSuccess) {
    fake_tm_.start_result = true;
    auto resp = roundtrip({{"cmd", "start_task"}, {"id", "cam0"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "cam0");
}

TEST_F(UnixSocketServerTest, StartTaskNotFound) {
    fake_tm_.start_result = false;
    auto resp = roundtrip({{"cmd", "start_task"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, StopTaskSuccess) {
    fake_tm_.stop_result = true;
    auto resp = roundtrip({{"cmd", "stop_task"}, {"id", "cam0"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "cam0");
}

TEST_F(UnixSocketServerTest, StopTaskNotFound) {
    fake_tm_.stop_result = false;
    auto resp = roundtrip({{"cmd", "stop_task"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, MetricsReturnsPrometheusText) {
    auto resp = roundtrip({{"cmd", "metrics"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_TRUE(resp["data"].is_string());
    EXPECT_FALSE(resp["data"].get<std::string>().empty());
}

TEST_F(UnixSocketServerTest, InvalidJsonReturnsError) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    ::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
    ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));

    std::string bad = "not valid json\n";
    ::send(fd, bad.data(), bad.size(), 0);

    std::string buf(4096, '\0');
    ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
    ::close(fd);

    buf.resize(n > 0 ? static_cast<size_t>(n) : 0);
    auto resp = json::parse(buf);
    EXPECT_EQ(resp["status"], "error");
}

TEST_F(UnixSocketServerTest, UnknownCommandReturnsError) {
    auto resp = roundtrip({{"cmd", "nonexistent"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

// ── list_sources ──────────────────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, ListSourcesEmpty) {
    auto resp = roundtrip({{"cmd", "list_sources"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_TRUE(resp["data"].is_array());
    EXPECT_TRUE(resp["data"].empty());
}

TEST_F(UnixSocketServerTest, ListSourcesWithItems) {
    fake_tm_.sources = {{"cam0", "rtsp://host/0", "STREAMING", 2},
                        {"cam1", "rtsp://host/1", "DEGRADED",  5}};
    auto resp = roundtrip({{"cmd", "list_sources"}});
    ASSERT_EQ(resp["data"].size(), 2u);
    EXPECT_EQ(resp["data"][0]["id"],              "cam0");
    EXPECT_EQ(resp["data"][0]["url"],             "rtsp://host/0");
    EXPECT_EQ(resp["data"][0]["state"],           "STREAMING");
    EXPECT_EQ(resp["data"][0]["reconnect_count"], 2);
    EXPECT_EQ(resp["data"][1]["state"],           "DEGRADED");
}

TEST_F(UnixSocketServerTest, ListSourcesStateIsString) {
    fake_tm_.sources = {{"cam0", "rtsp://x", "CONNECTING", 0}};
    auto resp = roundtrip({{"cmd", "list_sources"}});
    EXPECT_TRUE(resp["data"][0]["state"].is_string());
}

// ── list_pipelines ────────────────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, ListPipelinesEmpty) {
    auto resp = roundtrip({{"cmd", "list_pipelines"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_TRUE(resp["data"].is_array());
    EXPECT_TRUE(resp["data"].empty());
}

TEST_F(UnixSocketServerTest, ListPipelinesWithItems) {
    fake_tm_.pipelines = {{"pipe0", {"decode", "infer", "kafka"}, 2}};
    auto resp = roundtrip({{"cmd", "list_pipelines"}});
    ASSERT_EQ(resp["data"].size(), 1u);
    EXPECT_EQ(resp["data"][0]["id"],         "pipe0");
    EXPECT_EQ(resp["data"][0]["edge_count"], 2);
}

TEST_F(UnixSocketServerTest, ListPipelinesNodesIsArray) {
    fake_tm_.pipelines = {{"pipe0", {"a", "b"}, 1}};
    auto resp = roundtrip({{"cmd", "list_pipelines"}});
    EXPECT_TRUE(resp["data"][0]["nodes"].is_array());
    EXPECT_EQ(resp["data"][0]["nodes"].size(), 2u);
}

// ── list_models ───────────────────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, ListModelsEmpty) {
    auto resp = roundtrip({{"cmd", "list_models"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_TRUE(resp["data"].is_array());
    EXPECT_TRUE(resp["data"].empty());
}

TEST_F(UnixSocketServerTest, ListModelsWithItems) {
    fake_tm_.models = {{"yolov8n", "tensorrt", "v8", "3x640x640", 16, 1}};
    auto resp = roundtrip({{"cmd", "list_models"}});
    ASSERT_EQ(resp["data"].size(), 1u);
    EXPECT_EQ(resp["data"][0]["id"],             "yolov8n");
    EXPECT_EQ(resp["data"][0]["backend"],        "tensorrt");
    EXPECT_EQ(resp["data"][0]["batch_size"],     16);
    EXPECT_EQ(resp["data"][0]["instance_count"], 1);
}

TEST_F(UnixSocketServerTest, ListModelsInputShapeFormat) {
    fake_tm_.models = {{"m0", "onnx_cpu", "v8", "3x640x640", 1, 1}};
    auto resp = roundtrip({{"cmd", "list_models"}});
    const std::string shape = resp["data"][0]["input_shape"];
    EXPECT_NE(shape.find('x'), std::string::npos);
}

// ── add_source / remove_source ────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, AddSourceSuccess) {
    auto resp = roundtrip({{"cmd", "add_source"}, {"id", "cam_new"}, {"url", "rtsp://host/new"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "cam_new");
}

TEST_F(UnixSocketServerTest, AddSourceDuplicate) {
    fake_tm_.add_source_result = false;
    auto resp = roundtrip({{"cmd", "add_source"}, {"id", "cam_new"}, {"url", "rtsp://host/new"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, RemoveSourceSuccess) {
    auto resp = roundtrip({{"cmd", "remove_source"}, {"id", "cam_001"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "cam_001");
}

TEST_F(UnixSocketServerTest, RemoveSourceNotFound) {
    fake_tm_.remove_source_result = false;
    auto resp = roundtrip({{"cmd", "remove_source"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

// ── add_pipeline / remove_pipeline ───────────────────────────────────────────

TEST_F(UnixSocketServerTest, AddPipelineSuccess) {
    auto resp = roundtrip({
        {"cmd", "add_pipeline"},
        {"id",  "pipe_new"},
        {"nodes", json::array()},
        {"edges", json::array()}
    });
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "pipe_new");
}

TEST_F(UnixSocketServerTest, AddPipelineDuplicate) {
    fake_tm_.add_pipeline_result = false;
    auto resp = roundtrip({{"cmd", "add_pipeline"}, {"id", "pipe_dup"}, {"nodes", json::array()}, {"edges", json::array()}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, RemovePipelineSuccess) {
    auto resp = roundtrip({{"cmd", "remove_pipeline"}, {"id", "pipeline_default"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "pipeline_default");
}

TEST_F(UnixSocketServerTest, RemovePipelineNotFound) {
    fake_tm_.remove_pipeline_result = false;
    auto resp = roundtrip({{"cmd", "remove_pipeline"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

// ── add_task / remove_task ────────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, AddTaskSuccess) {
    auto resp = roundtrip({
        {"cmd",         "add_task"},
        {"id",          "task_new"},
        {"source_id",   "cam_001"},
        {"pipeline_id", "pipeline_default"}
    });
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "task_new");
}

TEST_F(UnixSocketServerTest, AddTaskMissingFields) {
    // Missing source_id and pipeline_id should return error.
    fake_tm_.add_task_result = false;
    auto resp = roundtrip({{"cmd", "add_task"}, {"id", "task_bad"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, RemoveTaskSuccess) {
    auto resp = roundtrip({{"cmd", "remove_task"}, {"id", "task_cam_001"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "task_cam_001");
}

TEST_F(UnixSocketServerTest, RemoveTaskNotFound) {
    fake_tm_.remove_task_result = false;
    auto resp = roundtrip({{"cmd", "remove_task"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

// ── load_model / unload_model ─────────────────────────────────────────────────

TEST_F(UnixSocketServerTest, LoadModelSuccess) {
    auto resp = roundtrip({
        {"cmd",     "load_model"},
        {"id",      "yolov8n_new"},
        {"backend", "cpu"},
        {"onnx_path", "/models/yolov8n.onnx"},
        {"batch_size", 1}
    });
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "yolov8n_new");
}

TEST_F(UnixSocketServerTest, LoadModelDuplicate) {
    fake_tm_.load_model_result = false;
    auto resp = roundtrip({{"cmd", "load_model"}, {"id", "dup"}, {"backend", "cpu"}, {"onnx_path", "/x"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}

TEST_F(UnixSocketServerTest, UnloadModelSuccess) {
    auto resp = roundtrip({{"cmd", "unload_model"}, {"id", "yolov8n"}});
    EXPECT_EQ(resp["status"], "ok");
    EXPECT_EQ(resp["id"],     "yolov8n");
}

TEST_F(UnixSocketServerTest, UnloadModelNotFound) {
    fake_tm_.unload_model_result = false;
    auto resp = roundtrip({{"cmd", "unload_model"}, {"id", "missing"}});
    EXPECT_EQ(resp["status"], "error");
    EXPECT_TRUE(resp.contains("message"));
}
