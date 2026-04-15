#include <gtest/gtest.h>
#include "common/Types.h"

using namespace infer;

// ── BBox ──────────────────────────────────────────────────────────────────

TEST(BBox, DefaultValues) {
    BBox b;
    EXPECT_FLOAT_EQ(b.x1, 0.f);
    EXPECT_FLOAT_EQ(b.y1, 0.f);
    EXPECT_FLOAT_EQ(b.x2, 0.f);
    EXPECT_FLOAT_EQ(b.y2, 0.f);
}

TEST(BBox, WidthHeightArea) {
    BBox b{10.f, 20.f, 50.f, 60.f};
    EXPECT_FLOAT_EQ(b.width(),  40.f);
    EXPECT_FLOAT_EQ(b.height(), 40.f);
    EXPECT_FLOAT_EQ(b.area(),   1600.f);
}

TEST(BBox, ZeroArea) {
    BBox b{5.f, 5.f, 5.f, 5.f};
    EXPECT_FLOAT_EQ(b.width(),  0.f);
    EXPECT_FLOAT_EQ(b.height(), 0.f);
    EXPECT_FLOAT_EQ(b.area(),   0.f);
}

TEST(BBox, NegativeWidthInvertedCoords) {
    // x2 < x1 → negative width; area() returns negative value
    BBox b{60.f, 0.f, 10.f, 40.f};
    EXPECT_LT(b.width(), 0.f);
    EXPECT_LT(b.area(),  0.f);
}

// ── Detection ─────────────────────────────────────────────────────────────

TEST(Detection, DefaultValues) {
    Detection d;
    EXPECT_EQ(d.class_id, 0);
    EXPECT_TRUE(d.class_name.empty());
    EXPECT_FLOAT_EQ(d.confidence, 0.f);
    EXPECT_TRUE(d.attributes.empty());
}

TEST(Detection, AttributesMap) {
    Detection d;
    d.attributes["vehicle_type"] = "sedan";
    d.attributes["color"]        = "red";
    EXPECT_EQ(d.attributes.at("vehicle_type"), "sedan");
    EXPECT_EQ(d.attributes.at("color"),        "red");
    EXPECT_EQ(d.attributes.size(), 2u);
}

// ── Batch ─────────────────────────────────────────────────────────────────

TEST(Batch, EmptyByDefault) {
    Batch b;
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(b.size(), 0);
    EXPECT_FALSE(b.is_gpu);
}

TEST(Batch, SizeReflectsMetas) {
    Batch b;
    b.metas.push_back(StreamMeta{});
    b.metas.push_back(StreamMeta{});
    EXPECT_EQ(b.size(), 2);
    EXPECT_FALSE(b.empty());
}

// ── InferResult ───────────────────────────────────────────────────────────

TEST(InferResult, DefaultValues) {
    InferResult r;
    EXPECT_FLOAT_EQ(r.frame_ts,   0.0);
    EXPECT_FLOAT_EQ(r.latency_ms, 0.0);
    EXPECT_TRUE(r.detections.empty());
    EXPECT_EQ(r.frame_seq, 0u);
}
