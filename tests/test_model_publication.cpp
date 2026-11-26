#include <gtest/gtest.h>

#include "pipeline/TaskManager.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <unistd.h>

namespace infer {
namespace {

namespace fs = std::filesystem;
const std::string kToken(32, 'a');
const std::string kAttempt(32, 'b');
const std::string kOtherToken(32, 'c');

class ModelPublicationTest : public testing::Test {
protected:
    void SetUp() override {
        auto pattern = (fs::temp_directory_path() / "infer-publication-XXXXXX").string();
        const char* dir = mkdtemp(pattern.data());
        ASSERT_NE(dir, nullptr);
        root_ = fs::canonical(dir);
        writeConfig();
    }

    void TearDown() override {
        if (!root_.empty()) fs::remove_all(root_);
    }

    std::unique_ptr<TaskManager> manager(TaskManager::ModelProbe probe) const {
        return std::make_unique<TaskManager>(loadConfig(configPath()), configPath(),
                                             std::unordered_map<std::string, IPublisher*>{},
                                             nullptr, std::move(probe));
    }

    std::string configPath() const { return (root_ / "server.yaml").string(); }

    void writeConfig(bool root_classifier = false) const {
        std::ofstream config(configPath());
        config << "server:\n  model_repository: \"" << root_.string() << "\"\n"
               << "publishers:\n  - id: test\n    type: kafka\n";
        if (root_classifier) {
            config << "models:\n  - id: classifier\n    backend: cpu\n"
                   << "    model_type: classifier\n    input_size: [96, 128]\n";
        }
    }

    fs::path stage(const std::string& id, const std::string& token = kToken, bool cascade = false) {
        const auto dir = root_ / ".uploads" / token / kAttempt / id;
        fs::create_directories(dir / "1");
        std::ofstream(dir / "1" / "weight.onnx") << "test weight";
        std::ofstream config(dir / "config.yaml");
        config << "backend: cpu\ninput_size: [96, 128]\nonnx_path: weight.onnx\n";
        if (cascade) config << "cascade:\n  - model_id: classifier\n    attribute_key: color\n";
        return dir;
    }

    fs::path root_;
};

TEST_F(ModelPublicationTest, FailedProbeKeepsCandidateAndRetryCommitsAndRegistersOnce) {
    const auto candidate = stage("vehicle");
    int checks = 0;
    auto tm = manager([&](const auto& dir) {
        EXPECT_EQ(dir, candidate);
        if (++checks == 1) throw std::runtime_error("cannot load weights");
    });

    const auto failure = tm->publishModel("vehicle", kToken, kAttempt);
    EXPECT_EQ(failure.code, "weight_check_failed");
    EXPECT_EQ(failure.message, "cannot load weights");
    EXPECT_TRUE(fs::exists(candidate));
    EXPECT_FALSE(fs::exists(root_ / "vehicle"));
    EXPECT_TRUE(tm->listModels().empty());

    EXPECT_TRUE(tm->publishModel("vehicle", kToken, kAttempt).code.empty());
    EXPECT_EQ(checks, 2);
    EXPECT_FALSE(fs::exists(candidate));
    EXPECT_TRUE(fs::exists(root_ / "vehicle" / "1" / "weight.onnx"));
    EXPECT_TRUE(tm->publishModel("vehicle", kToken, "").code.empty());
    EXPECT_EQ(checks, 2);
    const auto models = tm->listModels();
    ASSERT_EQ(models.size(), 1u);
    EXPECT_EQ(models[0].input_shape, "3x96x128");
    auto from_repo = scanModelRepository(root_.string());
    ASSERT_EQ(from_repo.size(), 1u);
    EXPECT_EQ(from_repo[0].onnx_path, (root_ / "vehicle" / "1" / "weight.onnx").string());
}

TEST_F(ModelPublicationTest, MissingCascadeDependencyLeavesRepositoryUnchanged) {
    const auto candidate = stage("vehicle", kToken, true);
    int checks = 0;
    auto tm = manager([&](const auto&) { ++checks; });

    EXPECT_EQ(tm->publishModel("vehicle", kToken, kAttempt).code, "missing_dependency");
    EXPECT_EQ(checks, 0);
    EXPECT_FALSE(fs::exists(root_ / "vehicle"));
    EXPECT_TRUE(fs::exists(candidate));

    ModelConfig classifier;
    classifier.id = "classifier";
    ASSERT_TRUE(tm->loadModel(classifier));
    const auto result = tm->publishModel("vehicle", kToken, kAttempt);
    EXPECT_EQ(result.code, "missing_dependency");
    EXPECT_NE(result.message.find("startup config or model repository"), std::string::npos);
    EXPECT_EQ(checks, 0);
    EXPECT_FALSE(fs::exists(root_ / "vehicle"));
    EXPECT_TRUE(fs::exists(candidate));
}

TEST_F(ModelPublicationTest, CascadeDependencyFromRootConfigSurvivesRestart) {
    writeConfig(true);
    stage("vehicle", kToken, true);
    int checks = 0;
    auto tm = manager([&](const auto&) { ++checks; });

    EXPECT_TRUE(tm->publishModel("vehicle", kToken, kAttempt).code.empty());
    EXPECT_EQ(checks, 1);
    EXPECT_EQ(manager({})->listModels().size(), 2u);
}

TEST_F(ModelPublicationTest, CascadeDependencyFromRepositorySurvivesRestart) {
    stage("vehicle", kToken, true);
    auto tm = manager([](const auto&) {});
    fs::rename(stage("classifier"), root_ / "classifier");
    ASSERT_TRUE(tm->loadFromRepository("classifier"));

    EXPECT_TRUE(tm->publishModel("vehicle", kToken, kAttempt).code.empty());
    EXPECT_EQ(manager({})->listModels().size(), 2u);
}

TEST_F(ModelPublicationTest, CascadeDependencyRemovedFromStartupConfigDuringProbeBlocksCommit) {
    writeConfig(true);
    const auto candidate = stage("vehicle", kToken, true);
    auto tm = manager([&](const auto&) { writeConfig(); });

    EXPECT_EQ(tm->publishModel("vehicle", kToken, kAttempt).code, "missing_dependency");
    EXPECT_TRUE(fs::exists(candidate));
    EXPECT_FALSE(fs::exists(root_ / "vehicle"));
}

TEST_F(ModelPublicationTest, IDAddedToStartupConfigBlocksPublicationBeforeProbe) {
    const auto candidate = stage("classifier");
    auto tm = manager([](const auto&) { ADD_FAILURE() << "conflicting upload reached the probe"; });
    writeConfig(true);

    EXPECT_EQ(tm->publishModel("classifier", kToken, kAttempt).code, "already_exists");
    EXPECT_TRUE(fs::exists(candidate));
    EXPECT_FALSE(fs::exists(root_ / "classifier"));
    EXPECT_TRUE(tm->listModels().empty());
    EXPECT_NO_THROW(loadConfig(configPath()));
}

TEST_F(ModelPublicationTest, IDAddedToStartupConfigDuringProbeBlocksCommit) {
    const auto candidate = stage("classifier");
    int checks = 0;
    auto tm = manager([&](const auto&) {
        ++checks;
        writeConfig(true);
    });

    EXPECT_EQ(tm->publishModel("classifier", kToken, kAttempt).code, "already_exists");
    EXPECT_EQ(checks, 1);
    EXPECT_TRUE(fs::exists(candidate));
    EXPECT_FALSE(fs::exists(root_ / "classifier"));
    EXPECT_TRUE(tm->listModels().empty());
    EXPECT_NO_THROW(loadConfig(configPath()));
}

TEST_F(ModelPublicationTest, ConcurrentUploadsCannotOverwriteSameModel) {
    stage("vehicle");
    stage("vehicle", kOtherToken);
    std::promise<void> entered;
    std::promise<void> resume;
    auto ready = entered.get_future();
    auto resumed = resume.get_future();
    auto tm = manager([&](const auto&) {
        entered.set_value();
        resumed.wait();
    });
    auto first = std::async(std::launch::async, [&] {
        return tm->publishModel("vehicle", kToken, kAttempt);
    });
    ready.wait();
    EXPECT_EQ(tm->publishModel("vehicle", kOtherToken, kAttempt).code, "publish_busy");
    EXPECT_TRUE(tm->listModels().empty());
    resume.set_value();
    EXPECT_TRUE(first.get().code.empty());
    EXPECT_EQ(tm->publishModel("vehicle", kOtherToken, kAttempt).code, "already_exists");
    EXPECT_EQ(tm->listModels().size(), 1u);
}

TEST_F(ModelPublicationTest, DetailsSeparateRepositoryVersionFromRegisteredConfig) {
    auto tm = manager({});
    fs::rename(stage("vehicle"), root_ / "vehicle");

    const auto repo_only = tm->describeModel("vehicle");
    ASSERT_TRUE(repo_only.model);
    EXPECT_FALSE(repo_only.registered);
    EXPECT_EQ(repo_only.repository.version, "1");
    EXPECT_EQ(repo_only.repository.weight_paths, std::vector<std::string>{"weight.onnx"});

    ASSERT_TRUE(tm->loadFromRepository("vehicle"));
    EXPECT_TRUE(tm->describeModel("vehicle").registered);
    ASSERT_EQ(tm->unloadModel("vehicle"), ITaskManager::UnregisterResult::Removed);
    EXPECT_FALSE(tm->describeModel("vehicle").registered);
    EXPECT_TRUE(fs::exists(root_ / "vehicle" / "1" / "weight.onnx"));

    ModelConfig config_only;
    config_only.id = "config_only";
    ASSERT_TRUE(tm->loadModel(config_only));
    EXPECT_TRUE(tm->describeModel("config_only").registered);
    EXPECT_EQ(tm->describeModel("config_only").repository.code, "model_not_found");
    EXPECT_FALSE(tm->describeModel("unknown").model);
}

TEST_F(ModelPublicationTest, DetailsCanDescribeRepositoryModelWhenExportIsUnavailable) {
    auto tm = manager({});
    fs::create_directory_symlink(stage("linked"), root_ / "linked");
    const auto detail = tm->describeModel("linked");
    ASSERT_TRUE(detail.model);
    EXPECT_EQ(detail.model->id, "linked");
    EXPECT_FALSE(detail.registered);
    EXPECT_EQ(detail.repository.code, "model_not_exportable");
}

TEST_F(ModelPublicationTest, ReferencesProtectStoppedTasksAndCascadeChildren) {
    AppConfig config;
    ModelConfig child;
    child.id = "child";
    ModelConfig parent;
    parent.id = "parent";
    parent.cascade.push_back({"child", {1}, 0.f, "color"});
    config.models = {child, parent};
    config.pipelines = {{"pipeline", {
        {"first", "infer.engine", {{"model_id", "parent"}}},
        {"second", "infer.engine", {{"model_id", "child"}}},
        {"third", "infer.engine", {{"model_id", "child"}}},
    }, {}}};
    config.tasks.push_back({"stopped", "source", "pipeline"});
    TaskManager tm(config, configPath(), {}, nullptr);

    const auto detail = tm.describeModel("child");
    EXPECT_EQ(detail.references.cascade_parents, std::vector<std::string>{"parent"});
    const auto& refs = detail.references.pipelines.at("pipeline");
    EXPECT_EQ(refs.direct_nodes, (std::vector<std::string>{"second", "third"}));
    EXPECT_EQ(refs.cascade_nodes, std::vector<std::string>{"first"});
    EXPECT_EQ(detail.tasks_by_pipeline.at("pipeline"), (std::vector<std::pair<std::string, TaskState>>{{"stopped", TaskState::Stopped}}));
    EXPECT_EQ(tm.unloadModel("child"), ITaskManager::UnregisterResult::InUse);
    EXPECT_EQ(tm.unloadModel("parent"), ITaskManager::UnregisterResult::InUse);
}

TEST_F(ModelPublicationTest, RetiredModelCannotAcquireNewReferences) {
    auto tm = manager({});
    ModelConfig child;
    child.id = "child";
    ASSERT_TRUE(tm->loadModel(child));
    ASSERT_EQ(tm->unloadModel("child"), ITaskManager::UnregisterResult::Removed);

    PipelineConfig pipeline{"new", {{"node", "infer.engine", {{"model_id", "child"}}}}, {}};
    EXPECT_FALSE(tm->addPipeline(pipeline));
    ASSERT_TRUE(tm->addPipeline({"editable", {}, {}}));
    pipeline.id = "editable";
    EXPECT_FALSE(tm->updatePipeline(pipeline));

    ModelConfig parent;
    parent.id = "parent";
    parent.cascade.push_back({"child", {}, 0.f, "attr"});
    EXPECT_FALSE(tm->loadModel(parent));
    ASSERT_TRUE(tm->loadModel(child));
    EXPECT_TRUE(tm->updatePipeline(pipeline));
    EXPECT_TRUE(tm->loadModel(parent));
    EXPECT_EQ(tm->unloadModel("child"), ITaskManager::UnregisterResult::InUse);
}

TEST_F(ModelPublicationTest, PreviewUsesRegisteredSnapshotAndRejectsConcurrentChecks) {
    auto config = loadConfig(configPath());
    std::promise<void> entered;
    std::promise<void> release;
    auto entered_future = entered.get_future();
    auto release_future = release.get_future();
    TaskManager tm(config, configPath(), {}, nullptr, [](const auto&) {},
                   [&](const ModelConfig& model, const std::string& image) {
                       EXPECT_EQ(model.id, "active");
                       EXPECT_EQ(image, "sample");
                       entered.set_value();
                       release_future.wait();
                       return nlohmann::json{{"status", "ok"}};
                   });
    auto model = ModelConfig{};
    model.id = "active";
    ASSERT_TRUE(tm.loadModel(model));
    EXPECT_EQ(tm.testModel("missing", "sample")["code"], "model_not_registered");
    auto test = std::async(std::launch::async, [&] { return tm.testModel("active", "sample"); });
    entered_future.wait();
    EXPECT_EQ(tm.testModel("active", "sample")["code"], "model_test_busy");
    EXPECT_EQ(tm.publishModel("unrelated", kToken, kAttempt).code, "publish_busy");
    release.set_value();
    EXPECT_EQ(test.get()["status"], "ok");
}

} // namespace
} // namespace infer
