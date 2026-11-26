#include "server/ModelTest.h"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace infer {
namespace {

class FakeImageBackend : public IInferBackend {
public:
    FakeImageBackend(int batch_size, std::vector<float> scores)
        : batch_size_(batch_size), scores_(std::move(scores)) {}

    void loadModel(const ModelConfig&) override {}
    void unloadModel() override {}
    bool isLoaded() const override { return true; }
    int minBatchSize() const override { return batch_size_; }
    int maxBatchSize() const override { return batch_size_; }
    DeviceType deviceType() const override { return DeviceType::CPU; }

    void infer(const Batch& batch, std::vector<float>& output) override {
        EXPECT_EQ(batch.size(), batch_size_);
        ASSERT_EQ(batch.frames.size(), static_cast<size_t>(batch_size_));
        for (const auto& frame : batch.frames) {
            EXPECT_EQ(frame.type(), CV_8UC3);
            EXPECT_EQ(frame.cols, 200);
            EXPECT_EQ(frame.rows, 100);
        }
        output = scores_;
    }

private:
    int batch_size_;
    std::vector<float> scores_;
};

TEST(ModelPreview, RescalesBoxesToDecodedImageAndUsesModelLabels) {
    ModelConfig config;
    config.input_shape = {1, 3, 50, 50};
    config.version = YOLOVersion::v26;
    config.num_classes = 2;
    config.class_names = {"person", "vehicle"};
    FakeImageBackend backend(1, {10, 5, 30, 25, 0.9f, 1, 0, 0, 15, 10, 0.1f, 0});

    auto result = evaluateModelImage(config, backend, cv::Mat::zeros(100, 200, CV_8UC3));
    EXPECT_EQ(result["width"], 200);
    EXPECT_EQ(result["height"], 100);
    EXPECT_TRUE(result["nms_thresh"].is_null());
    EXPECT_FALSE(result["preview_jpeg"].get<std::string>().empty());
    ASSERT_EQ(result["results"].size(), 1u);
    EXPECT_EQ(result["results"][0]["class_name"], "vehicle");
    EXPECT_EQ(result["results"][0]["bbox"], (nlohmann::json{{"x1", 40}, {"y1", 10}, {"x2", 120}, {"y2", 50}}));
}

TEST(ModelPreview, RanksOnlyFirstFrameOfFixedBatch) {
    ModelConfig config;
    config.model_type = ModelType::Classifier;
    config.conf_thresh = 0.5f;
    config.num_classes = 4;
    config.class_names = {"first", "second", "third", "fourth"};
    FakeImageBackend backend(2, {0.7f, 0.9f, 0.9f, 0.1f, 1.f, 1.f, 1.f, 1.f});

    auto result = evaluateModelImage(config, backend, cv::Mat::zeros(100, 200, CV_8UC3));
    ASSERT_EQ(result["results"].size(), 3u);
    EXPECT_EQ(result["results"][0]["class_name"], "second");
    EXPECT_EQ(result["results"][1]["class_name"], "third");
    EXPECT_EQ(result["results"][2]["class_name"], "first");
    EXPECT_FALSE(result["results"][0].contains("bbox"));
}

TEST(ModelPreview, HandlesEmptyDetectionsAndRejectsMalformedTensor) {
    ModelConfig config;
    config.version = YOLOVersion::v26;
    FakeImageBackend empty(1, {0, 0, 1, 1, 0.01f, 0});
    FakeImageBackend malformed(1, {0, 0, 1});
    const auto image = cv::Mat::zeros(100, 200, CV_8UC3);

    EXPECT_TRUE(evaluateModelImage(config, empty, image)["results"].empty());
    EXPECT_THROW(evaluateModelImage(config, malformed, image), std::runtime_error);
}

} // namespace
} // namespace infer
