#include <gtest/gtest.h>

#include "pipeline/stages/SahiSchedulerStage.h"
#include "pipeline/stages/SahiMergeStage.h"
#include "pipeline/stages/SahiRoiRegistry.h"

#include <opencv2/core.hpp>

namespace infer {
namespace {

EventEnvelope makeFrame(const std::string& stream_id, uint64_t seq, int w, int h) {
    auto frame = std::make_shared<Frame>();
    frame->image = cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    frame->meta.stream_id = stream_id;
    frame->meta.frame_seq = seq;
    frame->meta.orig_width = w;
    frame->meta.orig_height = h;
    EventEnvelope env;
    env.stream_id = stream_id;
    env.frame_seq = seq;
    env.frame = frame;
    env.event_id = stream_id + ":" + std::to_string(seq);
    return env;
}

struct EmittedTiles {
    std::vector<EventEnvelope> events;
    void operator()(const EventEnvelope& e) { events.push_back(e); }
};

// ---------------------------------------------------------------------------
// Fix 1: only tile #0 carries parent_frame
// ---------------------------------------------------------------------------
TEST(SahiScheduler, OnlyFirstTileCarriesParentFrame) {
    SahiSchedulerConfig cfg;
    cfg.tile_width = 960;
    cfg.tile_height = 1144;
    cfg.x_overlap_ratio = 0.0f;
    cfg.y_overlap_ratio = 0.0f;
    cfg.full_interval = 1;

    SahiSchedulerStage stage("test_pf", cfg);
    auto env = makeFrame("stream_pf", 1, 1920, 1144);
    EmittedTiles out;
    stage.process(env, std::ref(out));

    ASSERT_GE(out.events.size(), 2u);
    ASSERT_TRUE(out.events[0].sahi_tile.has_value());
    EXPECT_NE(out.events[0].sahi_tile->parent_frame, nullptr);

    for (std::size_t i = 1; i < out.events.size(); ++i) {
        ASSERT_TRUE(out.events[i].sahi_tile.has_value());
        EXPECT_EQ(out.events[i].sahi_tile->parent_frame, nullptr)
            << "tile " << i << " should NOT carry parent_frame";
    }
}

// ---------------------------------------------------------------------------
// Fix 2: ROI missing => single-tile fallback (not full grid) when throttled
// ---------------------------------------------------------------------------
TEST(SahiScheduler, EmptyRoiProducesSingleTileNotFullGrid) {
    SahiSchedulerConfig cfg;
    cfg.tile_width = 960;
    cfg.tile_height = 960;
    cfg.full_interval = 10;
    cfg.roi_max_age_frames = 100;
    cfg.fallback_full_min_gap_frames = 5;

    SahiSchedulerStage stage("test", cfg);

    SahiRoiRegistry::update("stream_a", 1, {});

    // Frame 1 (first frame) -> scheduled full pass via full_interval
    auto env1 = makeFrame("stream_a", 2, 7680, 1144);
    EmittedTiles out1;
    stage.process(env1, std::ref(out1));
    EXPECT_GT(out1.events.size(), 1u);

    // Frame 2 (not full_interval cadence) with empty ROI -> should produce exactly 1 tile
    auto env2 = makeFrame("stream_a", 3, 7680, 1144);
    EmittedTiles out2;
    stage.process(env2, std::ref(out2));
    EXPECT_EQ(out2.events.size(), 1u);
    if (!out2.events.empty()) {
        const auto& tile = out2.events[0].sahi_tile;
        ASSERT_TRUE(tile.has_value());
        EXPECT_EQ(tile->tile_x, 0);
        EXPECT_EQ(tile->tile_y, 0);
        EXPECT_EQ(tile->tile_width, 7680);
        EXPECT_EQ(tile->tile_height, 1144);
    }
}

TEST(SahiScheduler, FallbackFullPassIsThrottled) {
    SahiSchedulerConfig cfg;
    cfg.tile_width = 960;
    cfg.tile_height = 960;
    cfg.full_interval = 100;
    cfg.roi_max_age_frames = 1;
    cfg.fallback_full_min_gap_frames = 5;

    SahiSchedulerStage stage("test_throttle", cfg);

    int full_pass_count = 0;
    int single_tile_count = 0;
    for (int i = 0; i < 20; ++i) {
        auto env = makeFrame("stream_b", static_cast<uint64_t>(10 + i), 3840, 2160);
        EmittedTiles out;
        stage.process(env, std::ref(out));
        if (out.events.size() > 1) {
            full_pass_count++;
        } else if (out.events.size() == 1) {
            single_tile_count++;
        }
    }
    EXPECT_LE(full_pass_count, 6);
    EXPECT_GT(single_tile_count, 0);
}

// ---------------------------------------------------------------------------
// Fix 6: tile image is ROI submat (not clone) - shares data with parent
// ---------------------------------------------------------------------------
TEST(SahiScheduler, TileImageIsSubmatNotClone) {
    SahiSchedulerConfig cfg;
    cfg.tile_width = 960;
    cfg.tile_height = 1144;
    cfg.x_overlap_ratio = 0.0f;
    cfg.y_overlap_ratio = 0.0f;
    cfg.full_interval = 1;

    SahiSchedulerStage stage("test_submat", cfg);
    auto env = makeFrame("stream_submat", 1, 1920, 1144);
    EmittedTiles out;
    stage.process(env, std::ref(out));

    ASSERT_GE(out.events.size(), 2u);
    for (const auto& ev : out.events) {
        ASSERT_TRUE(ev.frame);
        EXPECT_FALSE(ev.frame->image.empty());
        // submat: datastart points into the parent's data region
        EXPECT_TRUE(ev.frame->image.isSubmatrix())
            << "tile image should be a submat (ROI), not a deep copy";
    }
}

// ---------------------------------------------------------------------------
// SahiMerge: parent_frame captured from tile #0
// ---------------------------------------------------------------------------
TEST(SahiMerge, CapturesParentFrameFromFirstTile) {
    SahiMergeConfig cfg;
    cfg.merge_iou = 0.55f;
    cfg.stale_timeout_ms = 5000;

    SahiMergeStage merge("test_merge", cfg);
    auto parent = std::make_shared<Frame>();
    parent->image = cv::Mat(1144, 7680, CV_8UC3, cv::Scalar(0));
    parent->meta.stream_id = "merge_stream";
    parent->meta.frame_seq = 100;

    std::vector<EventEnvelope> emitted;
    auto emit = [&](const EventEnvelope& e) { emitted.push_back(e); };

    // Tile 0: carries parent_frame
    {
        EventEnvelope tile;
        tile.stream_id = "merge_stream";
        tile.event_id = "merge_stream:100#tile0";
        auto tf = std::make_shared<Frame>();
        tf->image = cv::Mat(1144, 960, CV_8UC3);
        tf->meta = parent->meta;
        tile.frame = tf;
        tile.sahi_tile = SahiTileInfo{100, 1, 0, 0, 960, 1144, 7680, 1144, true, 2, parent};
        InferResult r;
        r.frame_seq = 1;
        r.stream_id = "merge_stream";
        tile.infer_result = r;
        merge.process(tile, emit);
    }
    EXPECT_TRUE(emitted.empty());

    // Tile 1: NO parent_frame
    {
        EventEnvelope tile;
        tile.stream_id = "merge_stream";
        tile.event_id = "merge_stream:100#tile1";
        auto tf = std::make_shared<Frame>();
        tf->image = cv::Mat(1144, 960, CV_8UC3);
        tf->meta = parent->meta;
        tile.frame = tf;
        tile.sahi_tile = SahiTileInfo{100, 2, 960, 0, 960, 1144, 7680, 1144, true, 2, nullptr};
        InferResult r;
        r.frame_seq = 2;
        r.stream_id = "merge_stream";
        tile.infer_result = r;
        merge.process(tile, emit);
    }
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_EQ(emitted[0].frame, parent) << "merged event should use the parent frame from tile #0";
    EXPECT_FALSE(emitted[0].sahi_tile.has_value()) << "merged event should not carry sahi_tile";
}

// ---------------------------------------------------------------------------
// SahiRoiRegistry: remove cleans up
// ---------------------------------------------------------------------------
TEST(SahiRoiRegistry, RemoveCleansEntry) {
    SahiRoiRegistry::update("cleanup_stream", 1, {});
    EXPECT_TRUE(SahiRoiRegistry::get("cleanup_stream").has_value());

    SahiRoiRegistry::remove("cleanup_stream");
    EXPECT_FALSE(SahiRoiRegistry::get("cleanup_stream").has_value());
}

} // namespace
} // namespace infer
