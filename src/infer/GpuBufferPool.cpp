#ifdef BUILD_TRT_BACKEND

#include "infer/GpuBufferPool.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <thread>
#include <chrono>

#define CUDA_POOL_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("[GpuBufferPool CUDA] ") \
                                 + cudaGetErrorString(_err)); \
    } \
} while(0)

namespace infer {

GpuBufferPool::GpuBufferPool(MemoryChecker checker)
    : memory_checker_(std::move(checker)) {
    if (!memory_checker_) {
        // Default: query cudaMemGetInfo; safe when < kOomGuardRatio
        memory_checker_ = [this]() -> bool {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0)
                return true; // query failed — allow through
            const float used_ratio = static_cast<float>(total_bytes - free_bytes)
                                     / static_cast<float>(total_bytes);
            Metrics::get().setGpuMemoryUsageRatio(static_cast<double>(used_ratio));
            return used_ratio < kOomGuardRatio;
        };
    }
}

bool GpuBufferPool::isSafe() const {
    return memory_checker_ ? memory_checker_() : true;
}

void GpuBufferPool::init(int device_id, int pool_size,
                         size_t input_bytes, size_t output_bytes) {
    device_id_ = device_id;
    CUDA_POOL_CHECK(cudaSetDevice(device_id_));

    slots_.resize(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        auto& s = slots_[i];
        s.idx    = i;
        s.in_use.store(false, std::memory_order_relaxed);
        CUDA_POOL_CHECK(cudaMalloc(&s.input_device,  input_bytes));
        CUDA_POOL_CHECK(cudaMalloc(&s.output_device, output_bytes));
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
    if (!isSafe()) {
        throw GpuMemoryPressureException("GPU memory usage above safety threshold");
    }

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
