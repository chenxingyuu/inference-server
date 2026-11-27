#include <gtest/gtest.h>

#include "pipeline/TaskManager.h"

#include <cstdio>
#include <future>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace infer {
namespace {

using namespace std::chrono_literals;

class LoopbackSocket {
public:
    ~LoopbackSocket() { if (fd >= 0) close(fd); }

    uint16_t bindPort() const {
        if (fd < 0) return 0;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t size = sizeof(address);
        if (bind(fd, reinterpret_cast<sockaddr*>(&address), size) != 0 ||
            getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) != 0) return 0;
        return ntohs(address.sin_port);
    }

    bool dropConnection() const {
        pollfd listener{fd, POLLIN, 0};
        if (poll(&listener, 1, 2000) <= 0) return false;
        const int peer = accept(fd, nullptr, nullptr);
        if (peer < 0) return false;
        close(peer);
        return true;
    }

    const int fd{socket(AF_INET, SOCK_STREAM, 0)};
};

template<class Predicate>
bool waitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

class SourceDiagnosticsTest : public testing::Test {
protected:
    void SetUp() override {
        AppConfig config;
        config.sources = {{"camera", "rtsp://fixture/camera"},
                          {"yard", "rtsp://fixture/yard"},
                          {"lobby", "rtsp://fixture/lobby"},
                          {"recording", "/videos/recording.mp4"}};
        path_ = "/tmp/infer-source-diagnostics-" + std::to_string(getpid()) + ".yaml";
        std::remove(runtimeStatePath(path_).c_str());
        init(std::move(config));
    }

    void TearDown() override {
        manager_.reset();
        std::remove(runtimeStatePath(path_).c_str());
    }

    void init(AppConfig config) {
        manager_.reset();
        manager_ = std::make_unique<TaskManager>(std::move(config), path_,
            std::unordered_map<std::string, IPublisher*>{}, nullptr);
    }

    void startCamera(uint16_t port) {
        AppConfig config;
        config.server.task_start_timeout_ms = 1000;
        PipelineSourceConfig camera;
        camera.id = "camera";
        camera.url = "rtsp://127.0.0.1:" + std::to_string(port) + "/camera";
        camera.open_timeout_ms = 3000;
        camera.max_reconnect_attempts = 0;
        config.sources.push_back(camera);
        PipelineConfig pipeline;
        pipeline.id = "ingest";
        pipeline.nodes.push_back({"input", "source.rtsp", {}});
        config.pipelines.push_back(std::move(pipeline));
        TaskConfig task;
        task.id = "live";
        task.source_id = "camera";
        task.pipeline_id = "ingest";
        config.tasks.push_back(std::move(task));
        init(std::move(config));
        manager_->loadAll();
        manager_->startAll();
    }

    bool startupTimeoutCleanedUp() {
        const auto tasks = manager_->listTasks();
        if (tasks.empty()) return false;
        const auto& task = tasks.front();
        return task.state == TaskState::Failed && !task.cleanup_pending &&
            task.failure && task.failure->code == "startup_timeout";
    }

    std::string path_;
    std::unique_ptr<TaskManager> manager_;
};

TEST_F(SourceDiagnosticsTest, UsesFirstFrameResultAndPreservesItOnServiceFailure) {
    const ITaskManager::SourceProbe passed = [](const PipelineSourceConfig&) {
        return SourceProbeCode::Passed;
    };
    const auto initial = manager_->listSources();
    EXPECT_EQ(initial.front().availability.status, "unchecked");
    EXPECT_EQ(initial.back().availability.status, "unsupported");
    EXPECT_EQ(manager_->probeSource("recording", passed).error, "source_probe_unsupported");
    EXPECT_EQ(manager_->probeSource("missing", passed).error, "source_not_found");

    const auto success = manager_->probeSource("camera", passed);
    ASSERT_TRUE(success.source);
    ASSERT_TRUE(success.source->last_probe);
    EXPECT_TRUE(success.source->last_probe->passed);
    EXPECT_EQ(success.source->availability.status, "probe_passed");
    EXPECT_EQ(success.source->availability.basis, "probe");

    const auto failure = manager_->probeSource("camera", [](const auto&) {
        return SourceProbeCode::AuthenticationFailed;
    });
    ASSERT_TRUE(failure.source);
    ASSERT_TRUE(failure.source->last_probe);
    EXPECT_FALSE(failure.source->last_probe->passed);
    EXPECT_EQ(failure.source->availability.reason, "authentication_failed");

    EXPECT_EQ(manager_->probeSource("camera", [](const auto&) -> SourceProbeCode {
        throw std::runtime_error("detector setup failed");
    }).error, "source_probe_unavailable");
    const auto sources = manager_->listSources();
    ASSERT_TRUE(sources.front().last_probe);
    EXPECT_EQ(sources.front().availability.status, "probe_failed");
    EXPECT_EQ(sources.front().last_probe->checked_at_ms, failure.source->last_probe->checked_at_ms);
    EXPECT_EQ(sources.front().state, "STOPPED");
}

TEST_F(SourceDiagnosticsTest, LimitsConcurrentProbesAndDiscardsResultsFromDeletedSource) {
    std::promise<void> release;
    const auto done = release.get_future().share();
    std::promise<void> camera_started, yard_started;
    auto camera = std::async(std::launch::async, [&] {
        return manager_->probeSource("camera", [&](const auto&) {
            camera_started.set_value();
            done.wait();
            return SourceProbeCode::Passed;
        });
    });
    auto yard = std::async(std::launch::async, [&] {
        return manager_->probeSource("yard", [&](const auto&) {
            yard_started.set_value();
            done.wait();
            return SourceProbeCode::Passed;
        });
    });
    camera_started.get_future().wait();
    yard_started.get_future().wait();

    const ITaskManager::SourceProbe unexpected = [](const PipelineSourceConfig&) {
        ADD_FAILURE() << "Probe callback should not run";
        return SourceProbeCode::Passed;
    };
    EXPECT_EQ(manager_->probeSource("camera", unexpected).error, "source_probe_in_progress");
    EXPECT_EQ(manager_->probeSource("lobby", unexpected).error, "source_probe_busy");
    EXPECT_TRUE(manager_->removeSource("camera"));
    EXPECT_TRUE(manager_->addSource({"camera", "rtsp://fixture/replacement"}));
    release.set_value();
    EXPECT_EQ(camera.get().error, "source_not_found");
    EXPECT_TRUE(yard.get().source);

    const auto sources = manager_->listSources();
    EXPECT_EQ(sources.back().id, "camera");
    EXPECT_EQ(sources.back().availability.status, "unchecked");
    EXPECT_FALSE(sources.back().last_probe);
}

TEST_F(SourceDiagnosticsTest, StartupTimeoutKeepsAnObservedRtspFailureAfterCleanup) {
    LoopbackSocket socket;
    const auto port = socket.bindPort();
    ASSERT_NE(port, 0);
    ASSERT_EQ(listen(socket.fd, 1), 0);
    auto dropped = std::async(std::launch::async, [&] { return socket.dropConnection(); });
    startCamera(port);
    ASSERT_TRUE(dropped.get());
    ASSERT_TRUE(waitFor([&] { return startupTimeoutCleanedUp(); }));
    const auto status = manager_->listSources().front().availability;
    EXPECT_EQ(status.status, "unavailable");
    EXPECT_EQ(status.basis, "runtime");
    EXPECT_EQ(status.reason, "stream_failed");
    EXPECT_TRUE(status.observed_at_ms);

    EXPECT_TRUE(manager_->stop("live"));
    EXPECT_EQ(manager_->listSources().front().availability.status, "unchecked");
}

TEST_F(SourceDiagnosticsTest, StartupTimeoutDoesNotBlameAnUnfinishedRtspHandshake) {
    LoopbackSocket socket;
    const auto port = socket.bindPort();
    ASSERT_NE(port, 0);
    // Leave RTSP unanswered after the TCP listener is ready.
    ASSERT_EQ(listen(socket.fd, 1), 0);
    startCamera(port);
    ASSERT_TRUE(waitFor([&] {
        return manager_->listSources().front().availability.status == "connecting";
    }));
    ASSERT_TRUE(waitFor([&] { return startupTimeoutCleanedUp(); }));
    EXPECT_EQ(manager_->listSources().front().availability.status, "unchecked");
}

TEST_F(SourceDiagnosticsTest, NewerProbeTakesPrecedenceOverRetainedStartupFailure) {
    LoopbackSocket socket;
    const auto port = socket.bindPort();
    ASSERT_NE(port, 0);
    ASSERT_EQ(listen(socket.fd, 1), 0);
    auto dropped = std::async(std::launch::async, [&] { return socket.dropConnection(); });
    startCamera(port);
    ASSERT_TRUE(dropped.get());
    ASSERT_TRUE(waitFor([&] {
        return manager_->listSources().front().availability.status == "reconnecting";
    }));
    ASSERT_TRUE(manager_->probeSource("camera", [](const auto&) { return SourceProbeCode::Passed; }).source);
    ASSERT_TRUE(waitFor([&] { return startupTimeoutCleanedUp(); }));
    EXPECT_EQ(manager_->listSources().front().availability.status, "probe_passed");
}

TEST(SourceObservationTest, RequiresNewFrameAfterReconnectAndKeepsInstancesIndependent) {
    using Phase = SourceObservation::Phase;
    SourceObservation first, second;
    first.transition(Phase::Connecting);
    first.transition(Phase::Open);
    EXPECT_FALSE(first.snapshot().last_frame);
    first.decoded();
    second.transition(Phase::Open);
    second.decoded();
    ASSERT_TRUE(first.snapshot().last_frame);
    ASSERT_TRUE(second.snapshot().last_frame);
    first.transition(Phase::Reconnecting);
    first.decoded();
    first.transition(Phase::Open);
    EXPECT_TRUE(first.snapshot().reconnecting_since);
    EXPECT_FALSE(first.snapshot().last_frame);
    first.transition(Phase::Idle);
    EXPECT_TRUE(second.snapshot().last_frame);
}

} // namespace
} // namespace infer
