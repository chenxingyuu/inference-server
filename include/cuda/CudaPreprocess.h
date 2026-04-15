#pragma once

#ifdef BUILD_TRT_BACKEND

#include <cuda_runtime.h>
#include <cstdint>

namespace infer::cuda {

// Fused NV12 → RGB CHW + bilinear resize + normalize in a single CUDA kernel.
//
// Input:   NV12 on GPU (Y plane + interleaved UV plane)
// Output:  float[C, dst_h, dst_w] in RGB channel order, values in [0, 1]
//
// All pointers must be device pointers.
// The function is async w.r.t. `stream` — callers must synchronize before
// reading dst_chw if they need a consistent result on the host.
void preprocessNV12ToCHW(
    const uint8_t* gpu_y,      // Y  plane: [src_h * src_w]
    const uint8_t* gpu_uv,     // UV plane: [src_h/2 * src_w] interleaved
    int src_h, int src_w,
    float*         dst_chw,    // output: [3, dst_h, dst_w]
    int dst_h, int dst_w,
    float mean,                // per-channel mean (same for R/G/B, typically 0)
    float scale,               // multiplier after mean subtraction (typically 1/255)
    cudaStream_t stream);

// Crop a bounding-box region from an NV12 frame and resize to dst_h×dst_w.
// Used by the cascade pipeline to prepare secondary classifier inputs.
// Coordinates are in absolute pixel space (not normalised).
void cropAndResizeNV12ToCHW(
    const uint8_t* gpu_y,
    const uint8_t* gpu_uv,
    int src_h, int src_w,
    int crop_x0, int crop_y0, int crop_x1, int crop_y1,
    float*         dst_chw,    // output: [3, dst_h, dst_w]
    int dst_h, int dst_w,
    float mean, float scale,
    cudaStream_t stream);

} // namespace infer::cuda

#endif // BUILD_TRT_BACKEND
