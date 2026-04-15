#include <gtest/gtest.h>
#include "decoder/IYOLODecoder.h"

using namespace infer;

// ── iou ───────────────────────────────────────────────────────────────────

TEST(Iou, IdenticalBoxes) {
    BBox a{0.f, 0.f, 10.f, 10.f};
    EXPECT_FLOAT_EQ(iou(a, a), 1.f);
}

TEST(Iou, NoOverlap) {
    BBox a{0.f, 0.f, 10.f, 10.f};
    BBox b{20.f, 20.f, 30.f, 30.f};
    EXPECT_FLOAT_EQ(iou(a, b), 0.f);
}

TEST(Iou, PartialOverlap) {
    // a: [0,0,10,10] area=100; b: [5,5,15,15] area=100; inter=[5,5,10,10]=25
    BBox a{0.f, 0.f, 10.f, 10.f};
    BBox b{5.f, 5.f, 15.f, 15.f};
    float expected = 25.f / (100.f + 100.f - 25.f);  // 25/175
    EXPECT_NEAR(iou(a, b), expected, 1e-5f);
}

TEST(Iou, ContainedBox) {
    // b fully inside a
    BBox a{0.f, 0.f, 10.f, 10.f};
    BBox b{2.f, 2.f,  8.f,  8.f};
    float inter = 6.f * 6.f;    // 36
    float area_a = 100.f;
    float area_b = 36.f;
    float expected = inter / (area_a + area_b - inter);  // 36/100
    EXPECT_NEAR(iou(a, b), expected, 1e-5f);
}

TEST(Iou, TouchingEdgeNoOverlap) {
    BBox a{0.f, 0.f, 10.f, 10.f};
    BBox b{10.f, 0.f, 20.f, 10.f};
    EXPECT_FLOAT_EQ(iou(a, b), 0.f);
}

// ── nonMaxSuppression ─────────────────────────────────────────────────────

static Detection makeDet(float conf, float x1, float y1, float x2, float y2) {
    Detection d;
    d.confidence = conf;
    d.bbox = {x1, y1, x2, y2};
    return d;
}

TEST(NMS, EmptyInput) {
    std::vector<Detection> dets;
    auto keep = nonMaxSuppression(dets, 0.5f);
    EXPECT_TRUE(keep.empty());
}

TEST(NMS, SingleDetection) {
    auto dets = std::vector<Detection>{makeDet(0.9f, 0.f, 0.f, 10.f, 10.f)};
    auto keep = nonMaxSuppression(dets, 0.5f);
    ASSERT_EQ(keep.size(), 1u);
    EXPECT_EQ(keep[0], 0);
}

TEST(NMS, SuppressesHighlyOverlapping) {
    // Two boxes with iou > threshold: only the higher-confidence one survives
    std::vector<Detection> dets = {
        makeDet(0.9f, 0.f, 0.f, 10.f, 10.f),
        makeDet(0.8f, 1.f, 1.f, 11.f, 11.f),  // heavily overlaps with first
    };
    auto keep = nonMaxSuppression(dets, 0.3f);
    ASSERT_EQ(keep.size(), 1u);
    EXPECT_EQ(keep[0], 0);  // higher confidence box survives
}

TEST(NMS, KeepsSeparateBoxes) {
    // Two boxes far apart: both survive
    std::vector<Detection> dets = {
        makeDet(0.9f,  0.f,  0.f, 10.f, 10.f),
        makeDet(0.8f, 50.f, 50.f, 60.f, 60.f),
    };
    auto keep = nonMaxSuppression(dets, 0.5f);
    EXPECT_EQ(keep.size(), 2u);
}

TEST(NMS, OrderedByConfidence) {
    // Lower-confidence box listed first — NMS must reorder by confidence
    std::vector<Detection> dets = {
        makeDet(0.5f, 0.f, 0.f, 10.f, 10.f),  // index 0 — lower conf
        makeDet(0.9f, 0.f, 0.f, 10.f, 10.f),  // index 1 — higher conf, identical box
    };
    auto keep = nonMaxSuppression(dets, 0.3f);
    ASSERT_EQ(keep.size(), 1u);
    EXPECT_EQ(keep[0], 1);  // the higher-confidence one wins
}

TEST(NMS, ZeroThresholdKeepsAllNonOverlapping) {
    std::vector<Detection> dets = {
        makeDet(0.9f,  0.f,  0.f, 10.f, 10.f),
        makeDet(0.8f, 20.f, 20.f, 30.f, 30.f),
        makeDet(0.7f, 40.f, 40.f, 50.f, 50.f),
    };
    auto keep = nonMaxSuppression(dets, 0.f);
    EXPECT_EQ(keep.size(), 3u);
}
