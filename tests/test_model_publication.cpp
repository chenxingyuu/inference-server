#include <gtest/gtest.h>

#include "pipeline/TaskManager.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>

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

} // namespace
} // namespace infer
