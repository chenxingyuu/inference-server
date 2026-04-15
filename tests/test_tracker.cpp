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
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::ByteTrack, 1, dets);

    ASSERT_EQ(dets.size(), 1u);
    ASSERT_TRUE(dets[0].track_id.has_value());
}

TEST(TrackerManager, ByteTrackKeepsStableIdForMatchedBox) {
    TrackerManager manager;
    std::vector<Detection> f1 = {makeDet(10, 20, 110, 220, 0.9f)};
    manager.apply("cam_1", TrackerType::ByteTrack, 1, f1);
    ASSERT_TRUE(f1[0].track_id.has_value());
    const int64_t id1 = f1[0].track_id.value();

    std::vector<Detection> f2 = {makeDet(12, 22, 112, 222, 0.92f)};
    manager.apply("cam_1", TrackerType::ByteTrack, 2, f2);
    ASSERT_TRUE(f2[0].track_id.has_value());
    EXPECT_EQ(f2[0].track_id.value(), id1);
}

TEST(TrackerManager, NoneLeavesDetectionsWithoutTrackId) {
    TrackerManager manager;
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::None, 1, dets);

    EXPECT_FALSE(dets[0].track_id.has_value());
}

TEST(TrackerManager, DeepSortPlaceholderDoesNotAssignTrackId) {
    TrackerManager manager;
    std::vector<Detection> dets = {makeDet(10, 20, 110, 220, 0.9f)};

    manager.apply("cam_1", TrackerType::DeepSort, 1, dets);

    EXPECT_FALSE(dets[0].track_id.has_value());
}
