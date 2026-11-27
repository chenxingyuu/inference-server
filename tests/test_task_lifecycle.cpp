#include <gtest/gtest.h>

#include "pipeline/TaskManager.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <unistd.h>

namespace infer {
namespace {

using namespace std::chrono_literals;

struct Signals {
    std::atomic<bool> emit_frame{false};
    std::atomic<bool> eof{false};
    std::atomic<bool> source_failed{false};
    std::atomic<bool> bad_start{false};
    std::atomic<bool> slow_start{false};
    std::atomic<bool> entered_start{false};
    std::atomic<bool> allow_output{true};
    std::atomic<bool> bad_process{false};
    std::atomic<int> processed{0};
    std::atomic<int> cleanups{0};
};

class ControlledInput final : public IStage {
public:
    explicit ControlledInput(Signals& signals) : signals_(signals) {}
    std::string id() const override { return "input"; }
    bool isSource() const override { return true; }
    bool inputEnded() const override { return signals_.eof.load(); }
    bool inputDrained() const override {
        return inputEnded() && (!signals_.emit_frame.load() || emitted_);
    }
    const char* terminalError() const override {
        return signals_.source_failed.load() ? "source_retries_exhausted" : nullptr;
    }
    void start() override {
        signals_.entered_start = true;
        while (signals_.slow_start.load()) std::this_thread::sleep_for(2ms);
        if (signals_.bad_start.load()) throw std::runtime_error("fixture startup exception");
    }
    void stop() override { ++signals_.cleanups; }
    void process(const EventEnvelope&, const EmitFn& emit) override {
        if (!signals_.emit_frame.load() || emitted_) return;
        emitted_ = true;
        EventEnvelope event;
        event.frame = std::make_shared<Frame>();
        emit(event);
    }
private:
    Signals& signals_;
    bool emitted_{false};
};

class ControlledOutput final : public IStage {
public:
    explicit ControlledOutput(Signals& signals) : signals_(signals) {}
    std::string id() const override { return "output"; }
    bool outputDrained() override { return signals_.allow_output.load(); }
    void process(const EventEnvelope&, const EmitFn&) override {
        if (signals_.bad_process.load()) throw std::runtime_error("fixture process exception");
        ++signals_.processed;
    }
private:
    Signals& signals_;
};

AppConfig config(int startup_timeout_ms) {
    AppConfig cfg;
    cfg.server.task_start_timeout_ms = startup_timeout_ms;
    cfg.sources.push_back({"shared-input", "unused"});
    PipelineConfig pipeline;
    pipeline.id = "pipeline";
    pipeline.edges = {EdgeConfig{"input", "output"}};
    cfg.pipelines.push_back(pipeline);
    for (const auto* id : {"one", "two"}) {
        TaskConfig task;
        task.id = id;
        task.source_id = "shared-input";
        task.pipeline_id = "pipeline";
        cfg.tasks.push_back(task);
    }
    return cfg;
}

template<class Predicate>
bool waitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(2ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

class TaskLifecycleTest : public testing::Test {
protected:
    void SetUp() override {
        path_ = (std::filesystem::temp_directory_path() /
            ("infer-lifecycle-" + std::to_string(getpid()) + "-" +
             testing::UnitTest::GetInstance()->current_test_info()->name())).string();
    }

    void TearDown() override {
        first_.slow_start = false;
        second_.slow_start = false;
        manager_.reset();
        std::filesystem::remove(runtimeStatePath(path_));
    }

    void init(int startup_timeout_ms = 1000, std::chrono::milliseconds drain_timeout = 1s) {
        manager_ = std::make_unique<TaskManager>(
            config(startup_timeout_ms), path_, std::unordered_map<std::string, IPublisher*>{}, nullptr,
            TaskManager::ModelProbe{}, TaskManager::ModelTest{},
            [this](const AppConfig& cfg, const TaskConfig& task) {
                auto pipeline = *cfg.findPipeline(task.pipeline_id);
                pipeline.id = task.id;
                auto graph = std::make_unique<GraphExecutor>(pipeline);
                auto& signals = task.id == "one" ? first_ : second_;
                graph->addStage(std::make_unique<ControlledInput>(signals));
                graph->addStage(std::make_unique<ControlledOutput>(signals));
                graph->build();
                return graph;
            }, drain_timeout);
        manager_->loadAll();
    }

    ITaskManager::TaskRuntimeInfo task(const std::string& id = "one") {
        for (const auto& task : manager_->listTasks()) {
            if (task.config.id == id) return task;
        }
        ADD_FAILURE() << "Missing task " << id;
        return {};
    }

    Signals first_, second_;
    std::string path_;
    std::unique_ptr<TaskManager> manager_;
};

TEST_F(TaskLifecycleTest, WaitsForFirstFrameAndIsolatesTasksWithSameSource) {
    init();
    manager_->startAll();
    EXPECT_EQ(task().state, TaskState::Starting);
    EXPECT_EQ(task("two").state, TaskState::Starting);
    first_.emit_frame = true;
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Running; }));
    EXPECT_EQ(task("two").state, TaskState::Starting);
    first_.source_failed = true;
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Failed; }));
    ASSERT_TRUE(task().failure);
    EXPECT_EQ(task().failure->code, "source_retries_exhausted");
    second_.emit_frame = true;
    ASSERT_TRUE(waitFor([&] { return task("two").state == TaskState::Running; }));
}

TEST_F(TaskLifecycleTest, SourceEditsWaitForStartingRunningAndDeletedTaskCleanup) {
    init(5000);
    SourceUpdate edit{};
    edit.url = "/videos/replacement.mp4";
    first_.slow_start = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return first_.entered_start.load(); }));
    EXPECT_EQ(manager_->updateSource("shared-input", edit).status, SourceUpdateStatus::Busy);
    first_.slow_start = false;
    first_.emit_frame = true;
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Running; }));
    EXPECT_EQ(manager_->updateSource("shared-input", edit).status, SourceUpdateStatus::Busy);
    ASSERT_TRUE(manager_->stop("one"));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Stopped && !task().cleanup_pending; }));

    second_.slow_start = true;
    ASSERT_TRUE(manager_->start("two"));
    ASSERT_TRUE(waitFor([&] { return second_.entered_start.load(); }));
    ASSERT_TRUE(manager_->removeTask("two"));
    EXPECT_EQ(manager_->updateSource("shared-input", edit).status, SourceUpdateStatus::Busy);
    second_.slow_start = false;
    ASSERT_TRUE(waitFor([&] {
        return manager_->updateSource("shared-input", edit).status == SourceUpdateStatus::Updated;
    }));
    EXPECT_EQ(manager_->listSources().front().url, edit.url);
    EXPECT_EQ(task().config.source_id, "shared-input");
}

TEST_F(TaskLifecycleTest, FailedStartupCleansUpAndAllowsRetry) {
    init();
    first_.bad_start = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Failed && !task().cleanup_pending; }));
    ASSERT_TRUE(task().failure);
    EXPECT_EQ(task().failure->code, "startup_failed");
    EXPECT_EQ(first_.cleanups.load(), 1);
    first_.bad_start = false;
    first_.emit_frame = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Running; }));
    EXPECT_FALSE(task().failure);
}

TEST_F(TaskLifecycleTest, TimeoutAndCancelledStartupIgnoreLateInput) {
    init(200);
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Failed; }));
    ASSERT_TRUE(task().failure);
    EXPECT_EQ(task().failure->code, "startup_timeout");

    ASSERT_TRUE(manager_->start("two"));
    ASSERT_TRUE(manager_->stop("two"));
    second_.emit_frame = true;
    ASSERT_TRUE(waitFor([&] { return task("two").state == TaskState::Stopped; }));
    EXPECT_FALSE(task("two").failure);
    ASSERT_TRUE(manager_->start("two"));
    ASSERT_TRUE(waitFor([&] { return task("two").state == TaskState::Running; }));
}

TEST_F(TaskLifecycleTest, TimeoutIsVisibleWhileStartupStillOwnsResources) {
    init(200);
    first_.slow_start = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return first_.entered_start.load(); }));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Failed; }));
    EXPECT_TRUE(task().cleanup_pending);
    EXPECT_EQ(manager_->requestStart("one"), ITaskManager::ActionResult::Busy);
    EXPECT_EQ(manager_->requestUpdate(task().config), ITaskManager::ActionResult::Busy);
    first_.slow_start = false;
    ASSERT_TRUE(waitFor([&] { return !task().cleanup_pending; }));
    EXPECT_EQ(first_.cleanups.load(), 1);
}

TEST_F(TaskLifecycleTest, DeletedTaskStaysHiddenUntilItsIdCanBeReused) {
    init();
    const auto config = task().config;
    first_.slow_start = true;
    ASSERT_TRUE(manager_->start(config.id));
    ASSERT_TRUE(waitFor([&] { return first_.entered_start.load(); }));
    ASSERT_TRUE(manager_->removeTask(config.id));
    ASSERT_EQ(manager_->listTasks().size(), 1);
    EXPECT_FALSE(manager_->addTask(config, false));
    first_.slow_start = false;
    ASSERT_TRUE(waitFor([&] { return manager_->addTask(config, false); }));
    EXPECT_EQ(task().state, TaskState::Stopped);
}

TEST_F(TaskLifecycleTest, ManualStopStaysStoppedUntilStartedAgain) {
    init();
    first_.emit_frame = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task("one").state == TaskState::Running; }));
    ASSERT_TRUE(manager_->stop("one"));
    ASSERT_TRUE(waitFor([&] { return task("one").state == TaskState::Stopped && !task("one").cleanup_pending; }));
    first_.emit_frame = false;

    manager_.reset();
    init();
    manager_->startAll();
    EXPECT_EQ(task("one").state, TaskState::Stopped);
    EXPECT_EQ(task("two").state, TaskState::Starting);

    ASSERT_TRUE(manager_->start("one"));
    EXPECT_EQ(task("one").state, TaskState::Starting);
    manager_.reset();
    init();
    manager_->startAll();
    EXPECT_EQ(task("one").state, TaskState::Starting);
}

TEST_F(TaskLifecycleTest, NaturalCompletionStillRestartsOnBoot) {
    init();
    first_.emit_frame = true;
    first_.eof = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task("one").state == TaskState::Stopped && !task("one").cleanup_pending; }));
    first_.emit_frame = false;
    first_.eof = false;

    manager_.reset();
    init();
    manager_->startAll();
    EXPECT_EQ(task("one").state, TaskState::Starting);
}

TEST_F(TaskLifecycleTest, FinishesTailBeforeStopping) {
    init();
    first_.allow_output = false;
    first_.emit_frame = true;
    first_.eof = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return first_.processed == 1 && task().state == TaskState::Running; }));
    first_.allow_output = true;
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Stopped; }));
    EXPECT_EQ(first_.cleanups.load(), 1);
    EXPECT_FALSE(task().failure);
}

TEST_F(TaskLifecycleTest, ReportsDrainTimeoutAndUnhandledStageException) {
    init(1000, 100ms);
    first_.allow_output = false;
    first_.emit_frame = true;
    first_.eof = true;
    ASSERT_TRUE(manager_->start("one"));
    ASSERT_TRUE(waitFor([&] { return task().state == TaskState::Failed; }));
    ASSERT_TRUE(task().failure);
    EXPECT_EQ(task().failure->code, "drain_timeout");

    second_.emit_frame = true;
    second_.bad_process = true;
    ASSERT_TRUE(manager_->start("two"));
    ASSERT_TRUE(waitFor([&] { return task("two").state == TaskState::Failed; }));
    ASSERT_TRUE(task("two").failure);
    EXPECT_EQ(task("two").failure->code, "pipeline_error");
}

} // namespace
} // namespace infer
