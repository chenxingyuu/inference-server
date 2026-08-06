#include <gtest/gtest.h>
#include "decoder/YOLO26Decoder.h"
#include "decoder/DecoderFactory.h"
#include "common/Config.h"
#include <vector>

using namespace infer;

// ── YOLO26Decoder ──────────────────────────────────────────────────────────
//
// YOLO26 is natively end-to-end (NMS-free). The exported model emits a fixed
// output layout: [batch, num_dets, 6]  (row-major, num_dets = 300 max).
//   Each row: [x1, y1, x2, y2, confidence, class_id]
//     - x1,y1,x2,y2 : absolute pixel corners (xyxy) in model-input resolution
//     - confidence  : detection score; padding rows are zero-filled
//     - class_id    : emitted as float, rounded to int
// No NMS is applied — overlapping high-confidence boxes are all kept.

static constexpr int kStride = 6;  // values per detection row

static std::vector<float> makeYolo26Output(int num_dets, int batch_size = 1) {
    return std::vector<float>(
        static_cast<size_t>(batch_size) * num_dets * kStride, 0.f);
}

// Place one detection row at index `det_idx` of image `batch_idx`.
static void placeYolo26Detection(std::vector<float>& buf,
                                 int batch_idx, int det_idx, int num_dets,
                                 float x1, float y1, float x2, float y2,
                                 float conf, float class_id) {
    float* p = buf.data() +
               (static_cast<size_t>(batch_idx) * num_dets + det_idx) * kStride;
    p[0] = x1;
    p[1] = y1;
    p[2] = x2;
    p[3] = y2;
    p[4] = conf;
    p[5] = class_id;
}

TEST(YOLO26Decoder, EmptyOutputNoPredictions) {
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(300);
    InferShape shape; shape.width = 640; shape.height = 640;

    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLO26Decoder, SingleDetectionDecodedFromXyxy) {
    const int num_dets = 300;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets);

    // xyxy corners are consumed directly (no cx/cy/wh conversion).
    placeYolo26Detection(buf, 0, 0, num_dets,
                         /*x1=*/270.f, /*y1=*/200.f,
                         /*x2=*/370.f, /*y2=*/280.f,
                         /*conf=*/0.91f, /*class_id=*/7.f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    const auto& det = results[0][0];
    EXPECT_EQ(det.class_id, 7);
    EXPECT_FLOAT_EQ(det.confidence, 0.91f);
    EXPECT_FLOAT_EQ(det.bbox.x1, 270.f);
    EXPECT_FLOAT_EQ(det.bbox.y1, 200.f);
    EXPECT_FLOAT_EQ(det.bbox.x2, 370.f);
    EXPECT_FLOAT_EQ(det.bbox.y2, 280.f);
}

TEST(YOLO26Decoder, LowConfidenceFilteredOut) {
    const int num_dets = 300;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets);

    placeYolo26Detection(buf, 0, 0, num_dets,
                         100.f, 100.f, 200.f, 200.f,
                         /*conf=*/0.2f, /*class_id=*/3.f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(YOLO26Decoder, ClassIdRoundedFromFloat) {
    const int num_dets = 300;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets);

    // Float class id that is not exactly integral must round to nearest int.
    placeYolo26Detection(buf, 0, 0, num_dets,
                         10.f, 10.f, 50.f, 50.f,
                         /*conf=*/0.8f, /*class_id=*/4.9999f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0].class_id, 5);
}

TEST(YOLO26Decoder, NmsFreeKeepsOverlappingBoxes) {
    // The defining YOLO26 behavior: predictions are already de-duplicated by
    // the end-to-end head, so the decoder must NOT run NMS. Two heavily
    // overlapping high-confidence boxes are both retained.
    const int num_dets = 300;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets);

    placeYolo26Detection(buf, 0, 0, num_dets,
                         270.f, 200.f, 370.f, 280.f, 0.95f, 1.f);
    placeYolo26Detection(buf, 0, 1, num_dets,
                         271.f, 201.f, 371.f, 281.f, 0.90f, 1.f);

    InferShape shape; shape.width = 640; shape.height = 640;
    // Even with a strict NMS threshold, both boxes survive (no NMS runs).
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), 2u);
}

TEST(YOLO26Decoder, CoordinatesClampedToImageBounds) {
    const int num_dets = 300;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets);

    // Box spills past every edge — must clamp to [0, w] x [0, h].
    placeYolo26Detection(buf, 0, 0, num_dets,
                         /*x1=*/-20.f, /*y1=*/-10.f,
                         /*x2=*/700.f, /*y2=*/680.f,
                         /*conf=*/0.9f, /*class_id=*/0.f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), 1, shape, 0.4f, 0.45f, buf.size());
    ASSERT_EQ(results[0].size(), 1u);
    const auto& bbox = results[0][0].bbox;
    EXPECT_FLOAT_EQ(bbox.x1, 0.f);
    EXPECT_FLOAT_EQ(bbox.y1, 0.f);
    EXPECT_FLOAT_EQ(bbox.x2, 640.f);
    EXPECT_FLOAT_EQ(bbox.y2, 640.f);
}

TEST(YOLO26Decoder, MultiBatchIndependent) {
    const int num_dets   = 300;
    const int batch_size = 2;
    YOLO26Decoder dec(80);
    auto buf = makeYolo26Output(num_dets, batch_size);

    // Image 0: one detection. Image 1: none.
    placeYolo26Detection(buf, 0, 0, num_dets,
                         100.f, 100.f, 200.f, 200.f, 0.9f, 2.f);

    InferShape shape; shape.width = 640; shape.height = 640;
    auto results = dec.decode(buf.data(), batch_size, shape, 0.4f, 0.45f,
                              buf.size());
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].size(), 1u);
    EXPECT_TRUE(results[1].empty());
}

TEST(YOLO26Decoder, Version) {
    YOLO26Decoder dec(80);
    EXPECT_EQ(dec.version(), YOLOVersion::v26);
}

TEST(YOLO26Decoder, FactoryCreatesV26) {
    ModelConfig cfg;
    cfg.model_type  = ModelType::Detector;
    cfg.version     = YOLOVersion::v26;
    cfg.num_classes = 80;
    auto dec = createDecoder(cfg);
    EXPECT_EQ(dec->version(), YOLOVersion::v26);
}
