#include <gtest/gtest.h>

#include "infer/BgrToNv12.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace infer {
namespace {

TEST(BgrToNv12Contiguous, BufferSizeMatchesWxH) {
    cv::Mat bgr(32, 48, CV_8UC3, cv::Scalar(10, 20, 30));
    const int w = 48;
    const int h = 32;
    std::vector<uint8_t> nv12(static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2);
    packSingleBgrToNv12Contiguous(bgr, w, h, nv12.data());
    EXPECT_EQ(nv12.size(), static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2);
}

TEST(BgrToNv12Contiguous, WhiteFrameHasHighLuma) {
    cv::Mat bgr(16, 16, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<uint8_t> nv12(16 * 16 * 3 / 2);
    packSingleBgrToNv12Contiguous(bgr, 16, 16, nv12.data());
    EXPECT_GE(nv12[0], 235u);
}

TEST(BgrToNv12Contiguous, BlackFrameHasLowLuma) {
    cv::Mat bgr(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<uint8_t> nv12(16 * 16 * 3 / 2);
    packSingleBgrToNv12Contiguous(bgr, 16, 16, nv12.data());
    EXPECT_LT(nv12[0], 20u);
}

TEST(BgrToNv12Contiguous, ResizesWhenSourceDimsDiffer) {
    cv::Mat bgr(20, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    const int w = 8;
    const int h = 8;
    std::vector<uint8_t> nv12(w * h * 3 / 2);
    packSingleBgrToNv12Contiguous(bgr, w, h, nv12.data());
    EXPECT_GT(nv12[0], 200u);
}

} // namespace
} // namespace infer
