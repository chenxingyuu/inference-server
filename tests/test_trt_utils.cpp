#include <gtest/gtest.h>
#include "infer/TrtUtils.h"

using namespace infer;

TEST(HasDynamicDims, EmptyDimsReturnsFalse) {
    EXPECT_FALSE(hasDynamicDims(0, nullptr));
}

TEST(HasDynamicDims, AllPositiveReturnsFalse) {
    int32_t d[] = {16, 3, 640, 640};
    EXPECT_FALSE(hasDynamicDims(4, d));
}

TEST(HasDynamicDims, DynamicBatchReturnsTrue) {
    int32_t d[] = {-1, 3, 640, 640};
    EXPECT_TRUE(hasDynamicDims(4, d));
}

TEST(HasDynamicDims, LastDimDynamicReturnsTrue) {
    int32_t d[] = {16, 3, 640, -1};
    EXPECT_TRUE(hasDynamicDims(4, d));
}

TEST(HasDynamicDims, AllDynamicReturnsTrue) {
    int32_t d[] = {-1, -1, -1, -1};
    EXPECT_TRUE(hasDynamicDims(4, d));
}

TEST(HasDynamicDims, NbDimsNegativeReturnsFalse) {
    // getTensorShape returns Dims with nbDims==-1 when tensor name not found.
    // hasDynamicDims must not iterate and must return false.
    int32_t d[] = {640, 640, 640, 640};
    EXPECT_FALSE(hasDynamicDims(-1, d));
}

TEST(HasDynamicDims, SingleStaticReturnsFalse) {
    int32_t d[] = {1};
    EXPECT_FALSE(hasDynamicDims(1, d));
}

TEST(HasDynamicDims, SingleDynamicReturnsTrue) {
    int32_t d[] = {-1};
    EXPECT_TRUE(hasDynamicDims(1, d));
}
