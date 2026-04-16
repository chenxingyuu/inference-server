#include <gtest/gtest.h>

#include "tracker/TrackerManager.h"

using namespace infer;

namespace {
Detection makeDet(float x1, float y1, float x2, float y2, float conf) {
    Detection d;
    d.class_id = 0;
    d.class_name = "person";
    d.confidence = conf;
    d.bbox = BBox{x1, y1, x2, y2};
    return d;
}
} // namespace

TEST(TrackerManager, ByteTrackAssignsTrackIds) {
    TrackerManager manager;
    ByteTrackConfig cfg;
    cfg.min_hits_to_confirm = 1;
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 1, dets);

    ASSERT_EQ(dets.size(), 1u);
    ASSERT_TRUE(dets[0].track_id.has_value());
}

TEST(TrackerManager, ByteTrackKeepsStableIdForMatchedBox) {
    TrackerManager manager;
    ByteTrackConfig cfg;
    cfg.min_hits_to_confirm = 1;
    std::vector<Detection> f1 = {makeDet(10, 20, 110, 220, 0.9f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 1, f1);
    ASSERT_TRUE(f1[0].track_id.has_value());
    const int64_t id1 = f1[0].track_id.value();

    std::vector<Detection> f2 = {makeDet(12, 22, 112, 222, 0.92f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 2, f2);
    ASSERT_TRUE(f2[0].track_id.has_value());
    EXPECT_EQ(f2[0].track_id.value(), id1);
}

TEST(TrackerManager, NoneLeavesDetectionsWithoutTrackId) {
    TrackerManager manager;
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::None, ByteTrackConfig{}, 1, dets);

    EXPECT_FALSE(dets[0].track_id.has_value());
}

TEST(TrackerManager, DeepSortPlaceholderDoesNotAssignTrackId) {
    TrackerManager manager;
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::DeepSort, ByteTrackConfig{}, 1, dets);

    EXPECT_FALSE(dets[0].track_id.has_value());
}

TEST(TrackerManager, ByteTrackPublishesIdOnlyAfterMinHitsToConfirm) {
    TrackerManager manager;
    ByteTrackConfig cfg;
    cfg.min_hits_to_confirm = 2;

    std::vector<Detection> f1 = {makeDet(10, 20, 110, 220, 0.9f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 1, f1);
    ASSERT_EQ(f1.size(), 1u);
    EXPECT_FALSE(f1[0].track_id.has_value());

    std::vector<Detection> f2 = {makeDet(11, 21, 111, 221, 0.91f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 2, f2);
    ASSERT_EQ(f2.size(), 1u);
    EXPECT_TRUE(f2[0].track_id.has_value());
}

TEST(TrackerManager, ByteTrackUsesGlobalAssignmentWhenLocalGreedyIsSuboptimal) {
    TrackerManager manager;
    ByteTrackConfig cfg;
    cfg.match_iou_thresh = 0.3f;
    cfg.min_hits_to_confirm = 1;

    std::vector<Detection> f1 = {
        makeDet(0, 0, 100, 100, 0.95f),
        makeDet(10, 0, 110, 100, 0.95f),
    };
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 1, f1);
    ASSERT_TRUE(f1[0].track_id.has_value());
    ASSERT_TRUE(f1[1].track_id.has_value());
    const int64_t id_left = f1[0].track_id.value();
    const int64_t id_right = f1[1].track_id.value();

    // d1 overlaps both old tracks heavily. d2 is only matchable to the first track.
    std::vector<Detection> f2 = {
        makeDet(5, 0, 105, 100, 0.95f),
        makeDet(-53, 0, 47, 100, 0.95f),
    };
    manager.apply("cam_1", TrackerType::ByteTrack, cfg, 2, f2);

    ASSERT_TRUE(f2[0].track_id.has_value());
    ASSERT_TRUE(f2[1].track_id.has_value());
    EXPECT_EQ(f2[0].track_id.value(), id_right);
    EXPECT_EQ(f2[1].track_id.value(), id_left);
}

TEST(TrackerManager, ByteTrackRecreatesTrackerWhenConfigChanges) {
    TrackerManager manager;
    ByteTrackConfig cfg1;
    cfg1.min_hits_to_confirm = 1;
    cfg1.match_iou_thresh = 0.3f;

    std::vector<Detection> f1 = {makeDet(0, 0, 100, 100, 0.9f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg1, 1, f1);
    ASSERT_TRUE(f1[0].track_id.has_value());
    EXPECT_EQ(f1[0].track_id.value(), 1);

    // With tighter IoU threshold, this displacement should fail matching.
    ByteTrackConfig cfg2 = cfg1;
    cfg2.match_iou_thresh = 0.9f;
    std::vector<Detection> f2 = {makeDet(30, 0, 130, 100, 0.9f)};
    manager.apply("cam_1", TrackerType::ByteTrack, cfg2, 2, f2);
    ASSERT_TRUE(f2[0].track_id.has_value());
    // Tracker recreation restarts stream-local ID allocation from 1.
    EXPECT_EQ(f2[0].track_id.value(), 1);
}
