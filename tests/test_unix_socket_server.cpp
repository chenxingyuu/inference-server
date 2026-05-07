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
    bool start_result{true};
    bool stop_result{true};

    std::vector<std::pair<std::string, State>> listTasks() const override { return tasks; }
    bool start(const std::string&) override { return start_result; }
    bool stop(const std::string&) override { return stop_result; }
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
