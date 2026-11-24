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
    }

    void TearDown() override {
        if (!root_.empty()) fs::remove_all(root_);
    }

    std::unique_ptr<TaskManager> manager(TaskManager::ModelProbe probe) const {
        AppConfig cfg;
        cfg.server.model_repository = root_.string();
        return std::make_unique<TaskManager>(std::move(cfg), (root_ / "server.yaml").string(),
                                             std::unordered_map<std::string, IPublisher*>{},
                                             nullptr, std::move(probe));
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
    EXPECT_TRUE(tm->publishModel("vehicle", kToken, kAttempt).code.empty());
    EXPECT_EQ(checks, 1);
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
