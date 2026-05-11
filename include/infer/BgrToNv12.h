#pragma once

#include "common/Types.h"

#include <cstddef>
#include <cstdint>

#include <opencv2/core/mat.hpp>

namespace infer {

// Pack one BGR uint8 frame (any WxH) into contiguous YUV420SP (NV12), size = W*H*3/2.
// Used by AscendBackend (AIPP path) and AscendVencFfmpegMuxWriter.
void packSingleBgrToNv12Contiguous(const cv::Mat& bgr, int width, int height, uint8_t* dst_nv12);

// Same layout as packSingleBgrToNv12Contiguous but for a Batch of CPU BGR frames resized to (w,h).
void packBgrBatchToNv12(const Batch& input, uint8_t* dst, int batch_size, int h, int w);

} // namespace infer
