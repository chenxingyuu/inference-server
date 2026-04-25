#include <gtest/gtest.h>
#include "stream/GpuDeviceAllocator.h"

using namespace infer;

TEST(GpuDeviceAllocatorTest, SingleDevice_AlwaysReturnsSameDevice) {
    GpuDeviceAllocator alloc({0});
    EXPECT_EQ(alloc.acquire(), 0);
    EXPECT_EQ(alloc.acquire(), 0);
}

TEST(GpuDeviceAllocatorTest, TwoDevices_LeastLoadedFirst) {
    GpuDeviceAllocator alloc({0, 1});
    // Both empty → pick first (device 0)
    int d0 = alloc.acquire();
    EXPECT_EQ(d0, 0);
    // Device 0 has 1, device 1 has 0 → pick device 1
    int d1 = alloc.acquire();
    EXPECT_EQ(d1, 1);
    // Both have 1 → pick first with min (device 0)
    int d2 = alloc.acquire();
    EXPECT_EQ(d2, 0);
}

TEST(GpuDeviceAllocatorTest, Release_DecreasesCount) {
    GpuDeviceAllocator alloc({0, 1});
    alloc.acquire(); // 0: count=1
    alloc.acquire(); // 1: count=1
    alloc.release(0); // 0: count=0
    // 0 is least loaded again
    EXPECT_EQ(alloc.acquire(), 0);
}

TEST(GpuDeviceAllocatorTest, ActiveStreams_ReflectsState) {
    GpuDeviceAllocator alloc({0, 1, 2});
    EXPECT_EQ(alloc.activeStreams(0), 0);
    EXPECT_EQ(alloc.activeStreams(1), 0);
    EXPECT_EQ(alloc.activeStreams(2), 0);

    alloc.acquire(); // 0
    alloc.acquire(); // 1
    alloc.acquire(); // 2
    alloc.acquire(); // 0
    EXPECT_EQ(alloc.activeStreams(0), 2);
    EXPECT_EQ(alloc.activeStreams(1), 1);
    EXPECT_EQ(alloc.activeStreams(2), 1);

    alloc.release(0);
    EXPECT_EQ(alloc.activeStreams(0), 1);
}

TEST(GpuDeviceAllocatorTest, EmptyDeviceList_Throws) {
    EXPECT_THROW(GpuDeviceAllocator alloc({}), std::invalid_argument);
}

TEST(GpuDeviceAllocatorTest, Release_NeverGoesNegative) {
    GpuDeviceAllocator alloc({0});
    // release without acquire — count stays at 0
    alloc.release(0);
    EXPECT_EQ(alloc.activeStreams(0), 0);
}

TEST(GpuDeviceAllocatorTest, ThreeDevices_FairnessUnderLoad) {
    GpuDeviceAllocator alloc({0, 1, 2});
    // Acquire 6 times — should round-robin across all three
    for (int i = 0; i < 6; ++i) alloc.acquire();
    EXPECT_EQ(alloc.activeStreams(0), 2);
    EXPECT_EQ(alloc.activeStreams(1), 2);
    EXPECT_EQ(alloc.activeStreams(2), 2);
}
