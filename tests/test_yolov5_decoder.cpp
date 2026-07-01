#include <gtest/gtest.h>
#include "decoder/YOLOv5Decoder.h"
#include "decoder/YOLOv8Decoder.h"
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

TEST(YOLOv5Decoder, PixelSpaceCoordsAscendExport) {
    // Ascend / ultralytics raw exports: cx,cy,w,h already in model-input pixels.
    const int num_classes = 46;
    const int num_preds   = 100;
    const int step        = 5 + num_classes;
    std::vector<float> buf(static_cast<size_t>(num_preds) * step, 0.f);

    float* p = buf.data();
    p[0] = 480.f;  // cx
    p[1] = 270.f;  // cy
    p[2] = 100.f;  // w
    p[3] = 80.f;   // h
    p[4] = 0.95f;
    p[5 + 3] = 0.95f;

    YOLOv5Decoder dec(num_classes);
    InferShape shape;
    shape.width  = 960;
    shape.height = 960;

    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    const auto& bbox = results[0][0].bbox;
    EXPECT_NEAR(bbox.x1, 430.f, 1.f);
    EXPECT_NEAR(bbox.y1, 230.f, 1.f);
    EXPECT_NEAR(bbox.x2, 530.f, 1.f);
    EXPECT_NEAR(bbox.y2, 310.f, 1.f);
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

// ── YOLOv8Decoder ─────────────────────────────────────────────────────────
//
// Output layout: [4 + num_classes, num_anchors]  (column-major by anchor)
// Box coords are cx/cy/w/h in pixel space (ultralytics raw ONNX export).

static std::vector<float> makeYolov8Output(int num_classes,
                                            int batch_size  = 1,
                                            int num_anchors = 8400) {
    const int rows = 4 + num_classes;
    return std::vector<float>(
        static_cast<size_t>(batch_size) * rows * num_anchors, 0.f);
}

// Place one anchor prediction at column `anchor_idx`.
// cx, cy, w, h are in pixels.
static void placeV8Prediction(std::vector<float>& buf,
                               int batch_idx, int anchor_idx,
                               int num_classes, int true_class,
                               float cx, float cy, float bw, float bh,
                               float cls_score,
                               int num_anchors = 8400) {
    const int rows   = 4 + num_classes;
    float*    base   = buf.data() +
                       static_cast<ptrdiff_t>(batch_idx) * rows * num_anchors;
    base[0 * num_anchors + anchor_idx] = cx;
    base[1 * num_anchors + anchor_idx] = cy;
    base[2 * num_anchors + anchor_idx] = bw;
    base[3 * num_anchors + anchor_idx] = bh;
    base[(4 + true_class) * num_anchors + anchor_idx] = cls_score;
}

TEST(YOLOv8Decoder, EmptyOutputNoPredictions) {
    YOLOv8Decoder dec(3);
    auto buf = makeYolov8Output(3);
    InferShape shape; shape.width = 640; shape.height = 640;

    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLOv8Decoder, CxCyWhConvertedToCorners) {
    // A box centered at (320,240) with size 100×80 should become
    // x1=270, y1=200, x2=370, y2=280
    const int num_classes = 2;
    YOLOv8Decoder dec(num_classes);
    auto buf = makeYolov8Output(num_classes);

    placeV8Prediction(buf, 0, 0, num_classes, /*class=*/1,
                      /*cx=*/320.f, /*cy=*/240.f,
                      /*bw=*/100.f, /*bh=*/ 80.f,
                      /*score=*/0.9f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.5f, 0.45f);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    const auto& bbox = results[0][0].bbox;
    EXPECT_FLOAT_EQ(bbox.x1, 270.f);  // cx - w/2
    EXPECT_FLOAT_EQ(bbox.y1, 200.f);  // cy - h/2
    EXPECT_FLOAT_EQ(bbox.x2, 370.f);  // cx + w/2
    EXPECT_FLOAT_EQ(bbox.y2, 280.f);  // cy + h/2
}

TEST(YOLOv8Decoder, X1LessThanX2AfterConversion) {
    // Sanity check: for any valid input, x1 < x2 and y1 < y2
    const int num_classes = 1;
    YOLOv8Decoder dec(num_classes);
    auto buf = makeYolov8Output(num_classes);

    placeV8Prediction(buf, 0, 0, num_classes, 0,
                      /*cx=*/381.f, /*cy=*/278.f,
                      /*bw=*/248.f, /*bh=*/160.f,
                      /*score=*/0.9f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.5f, 0.45f);

    ASSERT_EQ(results[0].size(), 1u);
    const auto& bbox = results[0][0].bbox;
    EXPECT_LT(bbox.x1, bbox.x2);
    EXPECT_LT(bbox.y1, bbox.y2);
}

TEST(YOLOv8Decoder, NmsSuppressesNearDuplicates) {
    // 11 nearly-identical boxes (mirroring the original bug report)
    const int num_classes = 5;
    YOLOv8Decoder dec(num_classes);
    auto buf = makeYolov8Output(num_classes);

    for (int i = 0; i < 11; ++i) {
        placeV8Prediction(buf, 0, i, num_classes, /*class=*/4,
                          /*cx=*/382.f + i * 0.1f, /*cy=*/278.f + i * 0.05f,
                          /*bw=*/248.f,             /*bh=*/160.f,
                          /*score=*/0.9f - i * 0.01f);
    }

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.5f, 0.45f);

    ASSERT_EQ(results.size(), 1u);
    // All 11 boxes heavily overlap → NMS should keep exactly 1
    EXPECT_EQ(results[0].size(), 1u);
}

TEST(YOLOv8Decoder, LowConfidenceFilteredOut) {
    const int num_classes = 1;
    YOLOv8Decoder dec(num_classes);
    auto buf = makeYolov8Output(num_classes);

    placeV8Prediction(buf, 0, 0, num_classes, 0,
                      320.f, 240.f, 100.f, 80.f, /*score=*/0.2f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.5f, 0.45f);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLOv8Decoder, MultiBatchIndependent) {
    const int num_classes = 1;
    const int batch_size  = 2;
    YOLOv8Decoder dec(num_classes);
    auto buf = makeYolov8Output(num_classes, batch_size);

    // Image 0: one detection
    placeV8Prediction(buf, 0, 0, num_classes, 0,
                      320.f, 240.f, 100.f, 80.f, 0.9f);
    // Image 1: all zeros → below threshold

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), batch_size, shape, 0.5f, 0.45f);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].size(), 1u);
    EXPECT_TRUE(results[1].empty());
}

TEST(YOLOv8Decoder, Version) {
    YOLOv8Decoder dec(80);
    EXPECT_EQ(dec.version(), YOLOVersion::v8);
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
