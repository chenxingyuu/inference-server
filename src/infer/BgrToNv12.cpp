#include "infer/BgrToNv12.h"

#include <cstring>

#include <opencv2/imgproc.hpp>

namespace infer {

void packSingleBgrToNv12Contiguous(const cv::Mat& bgr, int width, int height, uint8_t* dst_nv12) {
    cv::Mat resized;
    if (bgr.cols != width || bgr.rows != height) {
        cv::resize(bgr, resized, {width, height});
    } else {
        resized = bgr;
    }
    cv::Mat i420;
    cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t* i420_ptr = i420.ptr<uint8_t>(0);
    const size_t y_bytes = static_cast<size_t>(width) * static_cast<size_t>(height);

    std::memcpy(dst_nv12, i420_ptr, y_bytes);
    const uint8_t* u = i420_ptr + y_bytes;
    const uint8_t* v = u + (y_bytes / 4);
    uint8_t*       uv = dst_nv12 + y_bytes;
    for (size_t i = 0; i < (y_bytes / 4); ++i) {
        uv[2 * i + 0] = u[i];
        uv[2 * i + 1] = v[i];
    }
}

void packBgrBatchToNv12(const Batch& input, uint8_t* dst, int batch_size, int h, int w) {
    const size_t y_bytes       = static_cast<size_t>(h) * static_cast<size_t>(w);
    const size_t uv_bytes      = y_bytes / 2;
    const size_t frame_bytes   = y_bytes + uv_bytes;

    for (int b = 0; b < batch_size; ++b) {
        uint8_t* out = dst + static_cast<size_t>(b) * frame_bytes;
        packSingleBgrToNv12Contiguous(input.frames[static_cast<size_t>(b)], w, h, out);
    }
}

} // namespace infer
