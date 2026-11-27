#include <gtest/gtest.h>

#include "pipeline/TaskManager.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
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
    AppConfig baseConfig() {
        AppConfig config;
        config.sources = {{"camera", "rtsp://fixture/camera"},
                          {"yard", "rtsp://fixture/yard"},
                          {"lobby", "rtsp://fixture/lobby"},
                          {"recording", "/videos/recording.mp4"}};
        return config;
    }

    void SetUp() override {
        path_ = "/tmp/infer-source-diagnostics-" + std::to_string(getpid()) + ".yaml";
        std::remove(runtimeStatePath(path_).c_str());
        init(baseConfig());
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

    void configureCamera(uint16_t port) {
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
    }

    bool startupTimeoutCleanedUp() {
        const auto tasks = manager_->listTasks();
        if (tasks.empty()) return false;
        const auto& task = tasks.front();
        return task.state == TaskState::Failed && !task.cleanup_pending &&
            task.failure && task.failure->code == "startup_timeout";
    }

    std::string sourceStatus(const std::string& id) {
        for (const auto& source : manager_->listSources()) {
            if (source.id == id) return source.availability.status;
        }
        return {};
    }

    std::string path_;
    std::unique_ptr<TaskManager> manager_;
};

TEST_F(SourceDiagnosticsTest, EditsPersistForConfiguredAndAddedSources) {
    EXPECT_TRUE(manager_->addSource({"added", "/videos/old.mp4"}));
    SourceUpdate edit{};
    edit.url = "rtsp://fixture/new";
    edit.max_reconnect_attempts = 0;
    edit.open_timeout_ms = 2345;
    ASSERT_EQ(manager_->updateSource("camera", edit).status, SourceUpdateStatus::Updated);
    edit = {};
    edit.url = "/videos/new.mp4";
    ASSERT_EQ(manager_->updateSource("added", edit).status, SourceUpdateStatus::Updated);
    EXPECT_EQ(manager_->updateSource("missing", edit).status, SourceUpdateStatus::NotFound);
    EXPECT_EQ(manager_->updateSource("camera", edit).status, SourceUpdateStatus::InvalidArgument);

    init(baseConfig());
    const auto sources = manager_->listSources();
    EXPECT_EQ(sources.front().url, "rtsp://fixture/new");
    EXPECT_EQ(sources.front().config.max_reconnect_attempts, 0);
    EXPECT_EQ(sources.front().config.open_timeout_ms, 2345);
    EXPECT_EQ(sources.front().config.read_timeout_ms, 5000);
    EXPECT_EQ(sources.back().url, "/videos/new.mp4");
    EXPECT_EQ(sources.back().availability.status, "unsupported");
    ASSERT_TRUE(manager_->removeSource("camera"));
    ASSERT_TRUE(manager_->removeSource("added"));
    init(baseConfig());
    EXPECT_TRUE(sourceStatus("camera").empty());
    EXPECT_TRUE(sourceStatus("added").empty());
    ASSERT_TRUE(manager_->addSource({"camera", "rtsp://fixture/re-added"}));
    EXPECT_EQ(manager_->listSources().back().url, "rtsp://fixture/re-added");
}

TEST_F(SourceDiagnosticsTest, FailedSavePreservesConfigurationAndAvailability) {
    const auto probe = manager_->probeSource("camera", [](const auto&) { return SourceProbeCode::Passed; });
    ASSERT_TRUE(probe.source && probe.source->last_probe);
    // An existing directory at the destination prevents atomic replacement.
    ASSERT_TRUE(std::filesystem::create_directory(runtimeStatePath(path_)));
    SourceUpdate edit{};
    edit.url = "rtsp://fixture/fail";
    EXPECT_EQ(manager_->updateSource("camera", edit).status, SourceUpdateStatus::PersistenceFailed);
    const auto source = manager_->listSources().front();
    EXPECT_EQ(source.url, probe.source->url);
    EXPECT_EQ(source.availability.status, "online");
    ASSERT_TRUE(source.last_probe);
    EXPECT_EQ(source.last_probe->checked_at_ms, probe.source->last_probe->checked_at_ms);
}

TEST_F(SourceDiagnosticsTest, EditsResetProbeOnlyWhenItsSettingsChange) {
    const auto online = manager_->probeSource("camera", [](const auto&) { return SourceProbeCode::Passed; });
    ASSERT_TRUE(online.source && online.source->last_probe);
    SourceUpdate edit{};
    edit.url = online.source->url;
    edit.reconnect_delay_ms = 42;
    const auto reconnected = manager_->updateSource("camera", edit);
    ASSERT_TRUE(reconnected.source && reconnected.source->last_probe);
    EXPECT_EQ(reconnected.source->last_probe->checked_at_ms, online.source->last_probe->checked_at_ms);

    std::promise<void> started, release;
    const auto ready = started.get_future();
    const auto done = release.get_future();
    auto old_probe = std::async(std::launch::async, [&] {
        return manager_->probeSource("camera", [&](const auto&) {
            started.set_value();
            done.wait_for(2s);
            return SourceProbeCode::Passed;
        });
    });
    ASSERT_EQ(ready.wait_for(2s), std::future_status::ready);
    edit.read_timeout_ms = 3456;
    const auto changed = manager_->updateSource("camera", edit);
    ASSERT_TRUE(changed.source);
    EXPECT_EQ(changed.source->availability.status, "unchecked");
    EXPECT_FALSE(changed.source->last_probe);
    release.set_value();
    EXPECT_FALSE(old_probe.get().source);
    EXPECT_EQ(sourceStatus("camera"), "unchecked");
    manager_->startSourceMonitoring([](const auto& source) {
        return source.read_timeout_ms == 3456 ? SourceProbeCode::NoVideo : SourceProbeCode::Passed;
    });
    EXPECT_TRUE(waitFor([&] { return sourceStatus("camera") == "offline"; }));
}

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
    EXPECT_EQ(success.source->availability.status, "online");
    EXPECT_FALSE(success.reused_runtime);
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
    EXPECT_EQ(sources.front().availability.status, "offline");
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

TEST_F(SourceDiagnosticsTest, AutomaticallyTracksChangesAndRetriesAfterServiceFailure) {
    const auto result = std::make_shared<std::atomic<SourceProbeCode>>(SourceProbeCode::Passed);
    const auto unavailable_checks = std::make_shared<std::atomic<int>>(0);
    const auto file_checks = std::make_shared<std::atomic<int>>(0);
    manager_->startSourceMonitoring([result, unavailable_checks, file_checks](const auto& source) {
        if (source.id == "recording") ++*file_checks;
        const auto code = result->load();
        if (source.id == "camera" && code == SourceProbeCode::Unavailable) ++*unavailable_checks;
        return code;
    }, 20ms);

    EXPECT_TRUE(manager_->listTasks().empty());
    ASSERT_TRUE(waitFor([&] { return sourceStatus("camera") == "online"; }));
    result->store(SourceProbeCode::FirstFrameTimeout);
    ASSERT_TRUE(waitFor([&] { return sourceStatus("camera") == "offline"; }));
    const auto failed = manager_->listSources().front();
    ASSERT_TRUE(failed.last_probe);
    EXPECT_EQ(failed.availability.reason, "first_frame_timeout");

    result->store(SourceProbeCode::Unavailable);
    ASSERT_TRUE(waitFor([&] { return unavailable_checks->load() >= 2; }));
    const auto retained = manager_->listSources().front();
    ASSERT_TRUE(retained.last_probe);
    EXPECT_EQ(retained.availability.status, "offline");
    EXPECT_EQ(retained.availability.reason, "first_frame_timeout");

    result->store(SourceProbeCode::Passed);
    ASSERT_TRUE(waitFor([&] { return sourceStatus("camera") == "online"; }));
    EXPECT_EQ(sourceStatus("recording"), "unsupported");
    manager_->stopSourceMonitoring();
    EXPECT_EQ(file_checks->load(), 0);
}

TEST_F(SourceDiagnosticsTest, BackgroundAndManualChecksShareSlotsAndRejectStaleResults) {
    const auto started = std::make_shared<std::atomic<int>>(0);
    std::promise<void> release;
    const auto done = release.get_future().share();
    manager_->startSourceMonitoring([started, done](const auto& source) {
        if (source.url == "rtsp://fixture/replacement") return SourceProbeCode::AuthenticationFailed;
        if (source.id == "camera" || source.id == "yard") {
            ++*started;
            done.wait_for(2s);
        }
        return SourceProbeCode::Passed;
    });
    const bool both_started = waitFor([&] { return started->load() == 2; });
    if (!both_started) release.set_value();
    ASSERT_TRUE(both_started);
    EXPECT_EQ(sourceStatus("camera"), "unchecked");

    const ITaskManager::SourceProbe unexpected = [](const auto&) {
        ADD_FAILURE() << "Manual probe should not start while background slots are occupied";
        return SourceProbeCode::Passed;
    };
    EXPECT_EQ(manager_->probeSource("camera", unexpected).error, "source_probe_in_progress");
    EXPECT_EQ(manager_->probeSource("lobby", unexpected).error, "source_probe_busy");
    EXPECT_TRUE(manager_->removeSource("camera"));
    EXPECT_TRUE(manager_->addSource({"camera", "rtsp://fixture/replacement"}));
    EXPECT_EQ(manager_->probeSource("camera", unexpected).error, "source_probe_in_progress");
    EXPECT_EQ(sourceStatus("camera"), "unchecked");
    release.set_value();

    ASSERT_TRUE(waitFor([&] { return sourceStatus("camera") == "offline"; }));
    EXPECT_TRUE(waitFor([&] { return sourceStatus("lobby") == "online"; }));
    EXPECT_EQ(manager_->listSources().back().availability.reason, "authentication_failed");
}

TEST_F(SourceDiagnosticsTest, TaskStartFailureAndStopDoNotChangeAvailability) {
    LoopbackSocket socket;
    const auto port = socket.bindPort();
    ASSERT_NE(port, 0);
    ASSERT_EQ(listen(socket.fd, 1), 0);
    configureCamera(port);
    manager_->startSourceMonitoring([](const auto&) { return SourceProbeCode::Passed; });
    ASSERT_TRUE(waitFor([&] { return sourceStatus("camera") == "online"; }));
    const auto before_task = manager_->listSources().front().last_probe;
    ASSERT_TRUE(before_task);

    auto dropped = std::async(std::launch::async, [&] { return socket.dropConnection(); });
    ASSERT_TRUE(manager_->start("live"));
    ASSERT_TRUE(dropped.get());
    EXPECT_EQ(sourceStatus("camera"), "online");
    ASSERT_TRUE(waitFor([&] { return startupTimeoutCleanedUp(); }));
    EXPECT_EQ(sourceStatus("camera"), "online");
    EXPECT_TRUE(manager_->stop("live"));
    const auto after_task = manager_->listSources().front();
    EXPECT_EQ(after_task.availability.status, "online");
    ASSERT_TRUE(after_task.last_probe);
    EXPECT_EQ(after_task.last_probe->checked_at_ms, before_task->checked_at_ms);
}

} // namespace
} // namespace infer
