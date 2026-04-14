#ifdef BUILD_TRT_BACKEND

#include "cuda/CudaPreprocess.h"
#include <stdexcept>
#include <string>

namespace infer::cuda {

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("[CudaPreprocess] CUDA error: ") \
                                 + cudaGetErrorString(_err)); \
    } \
} while(0)

// Each thread processes one output pixel (r, c) for all 3 channels.
// NV12: Y plane is [src_h, src_w], UV plane is [src_h/2, src_w] interleaved.
// Bilinear resize from (src_h, src_w) → (dst_h, dst_w).
// Color conversion: NV12 (BT.601 limited range) → RGB [0, 255].
// Output: CHW float32, channel order R G B, values scaled by `scale` after
// subtracting `mean`.
__global__ void nv12ToRgbResizeNormKernel(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    int src_h, int src_w,
    float*  __restrict__ dst_chw,
    int dst_h, int dst_w,
    float mean, float scale)
{
    const int dst_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_col >= dst_w || dst_row >= dst_h) return;

    // Map destination pixel back to source coordinates (bilinear)
    const float fx = (dst_col + 0.5f) * (static_cast<float>(src_w) / dst_w) - 0.5f;
    const float fy = (dst_row + 0.5f) * (static_cast<float>(src_h) / dst_h) - 0.5f;

    const int x0 = max(0, static_cast<int>(fx));
    const int y0 = max(0, static_cast<int>(fy));
    const int x1 = min(src_w - 1, x0 + 1);
    const int y1 = min(src_h - 1, y0 + 1);

    const float wx1 = fx - x0;  // weight for x1
    const float wy1 = fy - y0;  // weight for y1
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    // Bilinear Y
    const float Y = wx0 * wy0 * y_plane[y0 * src_w + x0]
                  + wx1 * wy0 * y_plane[y0 * src_w + x1]
                  + wx0 * wy1 * y_plane[y1 * src_w + x0]
                  + wx1 * wy1 * y_plane[y1 * src_w + x1];

    // UV plane has half the vertical resolution; sample nearest UV row
    const int uv_row0 = (y0 / 2);
    const int uv_row1 = min(src_h / 2 - 1, uv_row0 + 1);
    const float uv_wy1 = fy / 2.0f - uv_row0;
    const float uv_wy0 = 1.0f - uv_wy1;

    // UV interleaved: U at [uv_row * src_w + x_even], V at [uv_row * src_w + x_even + 1]
    const int uv_col0 = (x0 / 2) * 2;  // even column for U
    const int uv_col1 = min(src_w - 2, uv_col0 + 2);

    const float U = wx0 * uv_wy0 * uv_plane[uv_row0 * src_w + uv_col0]
                  + wx1 * uv_wy0 * uv_plane[uv_row0 * src_w + uv_col1]
                  + wx0 * uv_wy1 * uv_plane[uv_row1 * src_w + uv_col0]
                  + wx1 * uv_wy1 * uv_plane[uv_row1 * src_w + uv_col1];

    const float V = wx0 * uv_wy0 * uv_plane[uv_row0 * src_w + uv_col0 + 1]
                  + wx1 * uv_wy0 * uv_plane[uv_row0 * src_w + uv_col1 + 1]
                  + wx0 * uv_wy1 * uv_plane[uv_row1 * src_w + uv_col0 + 1]
                  + wx1 * uv_wy1 * uv_plane[uv_row1 * src_w + uv_col1 + 1];

    // BT.601 full-range YCbCr → RGB
    const float Yc = Y  - 16.0f;
    const float Cb = U  - 128.0f;
    const float Cr = V  - 128.0f;

    float R = 1.164f * Yc + 1.596f * Cr;
    float G = 1.164f * Yc - 0.392f * Cb - 0.813f * Cr;
    float B = 1.164f * Yc + 2.017f * Cb;

    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    // Write CHW float: [R plane | G plane | B plane]
    const int pixel_idx = dst_row * dst_w + dst_col;
    const int plane     = dst_h * dst_w;
    dst_chw[0 * plane + pixel_idx] = (R - mean) * scale;
    dst_chw[1 * plane + pixel_idx] = (G - mean) * scale;
    dst_chw[2 * plane + pixel_idx] = (B - mean) * scale;
}

} // namespace

void preprocessNV12ToCHW(
    const uint8_t* gpu_y,
    const uint8_t* gpu_uv,
    int src_h, int src_w,
    float*         dst_chw,
    int dst_h, int dst_w,
    float mean, float scale,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y);

    nv12ToRgbResizeNormKernel<<<grid, block, 0, stream>>>(
        gpu_y, gpu_uv, src_h, src_w,
        dst_chw, dst_h, dst_w,
        mean, scale);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace infer::cuda

#endif // BUILD_TRT_BACKEND
