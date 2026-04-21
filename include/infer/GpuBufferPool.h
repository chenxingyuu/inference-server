#pragma once

#ifdef BUILD_TRT_BACKEND

#include <atomic>
#include <vector>
#include <cstddef>

namespace infer {

// A single pre-allocated GPU buffer slot containing input and output device memory.
// Used by TRTBackend to avoid per-inference cudaMalloc/cudaFree.
struct PooledBuffer {
    void* input_device{nullptr};   // CHW float:  batch * 3 * H * W * sizeof(float)
    void* output_device{nullptr};  // TRT output: batch * output_elems * sizeof(float)
    std::atomic<bool> in_use{false};
    int idx{-1};

    // Non-copyable, but movable (needed for vector::resize)
    PooledBuffer() = default;
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    PooledBuffer(PooledBuffer&& o) noexcept
        : input_device(o.input_device), output_device(o.output_device)
        , in_use(o.in_use.load(std::memory_order_relaxed)), idx(o.idx) {
        o.input_device = nullptr;
        o.output_device = nullptr;
        o.in_use.store(false, std::memory_order_relaxed);
    }
    PooledBuffer& operator=(PooledBuffer&&) = delete;
};

// Pre-allocated pool of GPU buffer pairs (input + output).
// Thread-safe: acquire()/release() use lock-free atomic CAS.
// Typical pool_size: 4 (double-buffering × 2 spare slots).
class GpuBufferPool {
public:
    GpuBufferPool() = default;
    ~GpuBufferPool() { reset(); }

    // Allocate pool_size slots on device_id. Must be called once before acquire().
    void init(int device_id, int pool_size,
              size_t input_bytes, size_t output_bytes);

    // Free all device memory. Safe to call multiple times.
    void reset();

    // Spin-acquire a free slot (yields between attempts, max 5ms).
    // Returns nullptr only on timeout (should not happen in normal operation).
    PooledBuffer* acquire();

    // Mark slot as available. Must be called exactly once per acquire().
    void release(PooledBuffer* buf);

    int poolSize() const { return static_cast<int>(slots_.size()); }

private:
    std::vector<PooledBuffer> slots_;
    int device_id_{0};
};

} // namespace infer

#endif // BUILD_TRT_BACKEND
