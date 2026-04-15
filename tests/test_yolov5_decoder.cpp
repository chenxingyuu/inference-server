#include <gtest/gtest.h>
#include "decoder/YOLOv5Decoder.h"
#include "decoder/ClassifierDecoder.h"
#include "decoder/DecoderFactory.h"
#include "common/Config.h"
#include <vector>
#include <cstring>

using namespace infer;

// ── Helpers ───────────────────────────────────────────────────────────────

// YOLOv5 output layout: [batch, 25200, 5+num_classes]
// Each row: [cx, cy, w, h, obj_conf, cls0..clsN-1]  (all in [0,1])
static std::vector<float> makeYolov5Output(int num_classes,
                                           int batch_size = 1) {
    const int num_preds = 25200;
    const int step      = 5 + num_classes;
    std::vector<float> buf(batch_size * num_preds * step, 0.f);
    return buf;
}

// Place a single high-confidence prediction at row `pred_idx` of image `b`.
static void placePrediction(std::vector<float>& buf,
                            int batch_idx, int pred_idx,
                            int num_classes, int true_class,
                            float cx, float cy, float bw, float bh,
                            float obj_conf, float cls_score) {
    const int num_preds = 25200;
    const int step      = 5 + num_classes;
    float* p = buf.data()
               + batch_idx * num_preds * step
               + pred_idx  * step;
    p[0] = cx;
    p[1] = cy;
    p[2] = bw;
    p[3] = bh;
    p[4] = obj_conf;
    p[5 + true_class] = cls_score;
}

// ── YOLOv5Decoder ─────────────────────────────────────────────────────────

TEST(YOLOv5Decoder, EmptyOutputNoPredictions) {
    YOLOv5Decoder dec(3);
    auto buf = makeYolov5Output(3);
    InferShape shape; shape.width = 640; shape.height = 640;

    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLOv5Decoder, SingleHighConfidencePrediction) {
    const int num_classes = 3;
    YOLOv5Decoder dec(num_classes);
    auto buf = makeYolov5Output(num_classes);

    // cx=0.5, cy=0.5, w=0.2, h=0.2, obj=0.95, cls1_score=0.95
    // best_prob = 0.95 * 0.95 = 0.9025 > thresh 0.4
    placePrediction(buf, 0, 0, num_classes,
                    /*true_class=*/1,
                    /*cx=*/0.5f, /*cy=*/0.5f,
                    /*bw=*/0.2f, /*bh=*/0.2f,
                    /*obj_conf=*/0.95f, /*cls_score=*/0.95f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0].class_id, 1);
    EXPECT_GT(results[0][0].confidence, 0.4f);

    // bbox should be clamped to [0, 640]
    EXPECT_GE(results[0][0].bbox.x1, 0.f);
    EXPECT_LE(results[0][0].bbox.x2, 640.f);
    EXPECT_GE(results[0][0].bbox.y1, 0.f);
    EXPECT_LE(results[0][0].bbox.y2, 640.f);
}

TEST(YOLOv5Decoder, LowConfidenceFilteredOut) {
    const int num_classes = 3;
    YOLOv5Decoder dec(num_classes);
    auto buf = makeYolov5Output(num_classes);

    // obj=0.1, cls_score=0.1 → best_prob=0.01, below thresh 0.4
    placePrediction(buf, 0, 0, num_classes, 0,
                    0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.1f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLOv5Decoder, NmsSuppressesDuplicates) {
    const int num_classes = 1;
    YOLOv5Decoder dec(num_classes);
    auto buf = makeYolov5Output(num_classes);

    // Two nearly-identical boxes — NMS should keep only one
    placePrediction(buf, 0, 0, num_classes, 0,
                    0.5f, 0.5f, 0.3f, 0.3f, 0.95f, 0.95f);
    placePrediction(buf, 0, 1, num_classes, 0,
                    0.51f, 0.51f, 0.3f, 0.3f, 0.90f, 0.90f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.3f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), 1u);
}

TEST(YOLOv5Decoder, MultiBatchIndependent) {
    const int num_classes = 2;
    const int batch_size  = 2;
    YOLOv5Decoder dec(num_classes);
    auto buf = makeYolov5Output(num_classes, batch_size);

    // Image 0: one detection (class 0)
    placePrediction(buf, 0, 0, num_classes, 0,
                    0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0.9f);
    // Image 1: no detection (all zeros → below thresh)

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), batch_size, shape, 0.4f, 0.45f);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].size(), 1u);   // image 0 has 1 det
    EXPECT_TRUE(results[1].empty());    // image 1 has 0 det
}

TEST(YOLOv5Decoder, Version) {
    YOLOv5Decoder dec(80);
    EXPECT_EQ(dec.version(), YOLOVersion::v5);
}

// ── ClassifierDecoder ─────────────────────────────────────────────────────

static ModelConfig makeClassifierCfg(int num_classes,
                                     std::vector<std::string> names = {}) {
    ModelConfig cfg;
    cfg.model_type  = ModelType::Classifier;
    cfg.num_classes = num_classes;
    cfg.class_names = std::move(names);
    return cfg;
}

TEST(ClassifierDecoder, TopOneAboveThreshold) {
    auto cfg = makeClassifierCfg(3, {"cat", "dog", "bird"});
    ClassifierDecoder dec(cfg);

    // batch=1, scores: [0.1, 0.8, 0.3] → class 1 "dog" wins
    std::vector<float> scores = {0.1f, 0.8f, 0.3f};
    InferShape shape;
    auto results = dec.decode(scores.data(), 1, shape, 0.5f, 0.f);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0].class_id,   1);
    EXPECT_EQ(results[0][0].class_name, "dog");
    EXPECT_FLOAT_EQ(results[0][0].confidence, 0.8f);
}

TEST(ClassifierDecoder, BelowThresholdFiltered) {
    auto cfg = makeClassifierCfg(3, {"cat", "dog", "bird"});
    ClassifierDecoder dec(cfg);

    std::vector<float> scores = {0.1f, 0.3f, 0.2f};  // all below 0.5
    InferShape shape;
    auto results = dec.decode(scores.data(), 1, shape, 0.5f, 0.f);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(ClassifierDecoder, MultiBatch) {
    auto cfg = makeClassifierCfg(2, {"neg", "pos"});
    ClassifierDecoder dec(cfg);

    // batch=2: [0.2, 0.9] and [0.8, 0.1]
    std::vector<float> scores = {0.2f, 0.9f,   // sample 0 → class 1
                                 0.8f, 0.1f};  // sample 1 → class 0
    InferShape shape;
    auto results = dec.decode(scores.data(), 2, shape, 0.5f, 0.f);

    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0].class_id, 1);
    ASSERT_EQ(results[1].size(), 1u);
    EXPECT_EQ(results[1][0].class_id, 0);
}

TEST(ClassifierDecoder, FallbackClassNameWhenNotProvided) {
    auto cfg = makeClassifierCfg(3);  // no class_names
    ClassifierDecoder dec(cfg);

    std::vector<float> scores = {0.1f, 0.9f, 0.2f};
    InferShape shape;
    auto results = dec.decode(scores.data(), 1, shape, 0.5f, 0.f);

    ASSERT_EQ(results[0].size(), 1u);
    // No class_names → class_name is stringified class_id
    EXPECT_EQ(results[0][0].class_name, "1");
}

TEST(ClassifierDecoder, ZeroClassesThrows) {
    ModelConfig cfg;
    cfg.model_type  = ModelType::Classifier;
    cfg.num_classes = 0;
    EXPECT_THROW(ClassifierDecoder dec(cfg), std::runtime_error);
}

// ── DecoderFactory ────────────────────────────────────────────────────────

TEST(DecoderFactory, CreatesCorrectDecoderForVersion) {
    auto makeDetCfg = [](YOLOVersion v) {
        ModelConfig cfg;
        cfg.model_type  = ModelType::Detector;
        cfg.version     = v;
        cfg.num_classes = 80;
        return cfg;
    };

    EXPECT_EQ(createDecoder(makeDetCfg(YOLOVersion::v5))->version(),  YOLOVersion::v5);
    EXPECT_EQ(createDecoder(makeDetCfg(YOLOVersion::v8))->version(),  YOLOVersion::v8);
    EXPECT_EQ(createDecoder(makeDetCfg(YOLOVersion::v11))->version(), YOLOVersion::v11);
    EXPECT_EQ(createDecoder(makeDetCfg(YOLOVersion::v26))->version(), YOLOVersion::v26);
}

TEST(DecoderFactory, CreatesClassifierDecoder) {
    ModelConfig cfg;
    cfg.model_type  = ModelType::Classifier;
    cfg.num_classes = 5;
    auto dec = createDecoder(cfg);
    EXPECT_EQ(dec->version(), YOLOVersion::Unknown);
}
