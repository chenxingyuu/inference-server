#ifdef BUILD_TRT_BACKEND

#include "infer/GpuBufferPool.h"
#include "common/Logger.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <thread>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("[GpuBufferPool CUDA] ") \
                                 + cudaGetErrorString(_err)); \
    } \
} while(0)

namespace infer {

void GpuBufferPool::init(int device_id, int pool_size,
                         size_t input_bytes, size_t output_bytes) {
    device_id_ = device_id;
    CUDA_CHECK(cudaSetDevice(device_id_));

    slots_.resize(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        auto& s = slots_[i];
        s.idx    = i;
        s.in_use.store(false, std::memory_order_relaxed);
        CUDA_CHECK(cudaMalloc(&s.input_device,  input_bytes));
        CUDA_CHECK(cudaMalloc(&s.output_device, output_bytes));
    }

    LOG_INFO("GpuBufferPool: allocated {} slots ({} + {} bytes each)",
             pool_size, input_bytes, output_bytes);
}

void GpuBufferPool::reset() {
    if (slots_.empty()) return;
    cudaSetDevice(device_id_);
    for (auto& s : slots_) {
        if (s.input_device)  { cudaFree(s.input_device);  s.input_device  = nullptr; }
        if (s.output_device) { cudaFree(s.output_device); s.output_device = nullptr; }
        s.in_use.store(false, std::memory_order_relaxed);
    }
    slots_.clear();
}

PooledBuffer* GpuBufferPool::acquire() {
    // Spin with back-off: try each slot in round-robin.
    // Inference passes are short (< few ms) so contention is rare.
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(5);

    while (std::chrono::steady_clock::now() < deadline) {
        for (auto& s : slots_) {
            bool expected = false;
            if (s.in_use.compare_exchange_weak(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
                return &s;
            }
        }
        std::this_thread::yield();
    }

    LOG_WARN("GpuBufferPool: acquire() timed out after 5ms — pool exhausted");
    return nullptr;
}

void GpuBufferPool::release(PooledBuffer* buf) {
    if (!buf) return;
    buf->in_use.store(false, std::memory_order_release);
}

} // namespace infer

#endif // BUILD_TRT_BACKEND
