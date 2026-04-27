#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include <acl/acl.h>
#include <atomic>
#include <functional>
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace infer {

// Thrown when NPU HBM memory usage exceeds the safety threshold.
struct AscendMemoryPressureException : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Pre-allocated NPU buffer slot: input + output HBM device memory.
// Used by AscendBackend to avoid per-inference aclrtMalloc/aclrtFree.
struct AscendPooledBuffer {
    void* input_device{nullptr};   // CHW float or raw BGR uint8 (AIPP path)
    void* output_device{nullptr};  // model output floats
    std::atomic<bool> in_use{false};
    int idx{-1};

    AscendPooledBuffer() = default;
    AscendPooledBuffer(const AscendPooledBuffer&) = delete;
    AscendPooledBuffer& operator=(const AscendPooledBuffer&) = delete;
    AscendPooledBuffer(AscendPooledBuffer&&) noexcept;
    AscendPooledBuffer& operator=(AscendPooledBuffer&&) = delete;
};

// Lock-free CAS pool of pre-allocated NPU buffer pairs (input + output).
// Mirrors GpuBufferPool design; substitutes cudaMalloc with aclrtMalloc.
// Typical pool_size: 4 (double-buffering × 2 spare slots).
class AscendBufferPool {
public:
    // Optional memory safety checker: returns true when safe to allocate.
    // Defaults to aclrtGetMemInfo-based 85% threshold check.
    // Inject a custom checker in tests to avoid requiring real NPU hardware.
    using MemoryChecker = std::function<bool()>;

    explicit AscendBufferPool(MemoryChecker checker = {});
    ~AscendBufferPool() { reset(); }

    // Allocate pool_size slots. Must be called once before acquire().
    // ctx must be the already-current ACL context for device_id; init() uses it
    // for aclrtMalloc and stores it so reset() can rebind before aclrtFree.
    void init(int device_id, aclrtContext ctx, int pool_size,
              size_t input_bytes, size_t output_bytes);

    // Test hook: allocates with ::operator new instead of aclrtMalloc.
    // Enabled when BUILD_ASCEND_BACKEND is defined; no real NPU needed.
    void initForTest(int pool_size, size_t input_bytes, size_t output_bytes);

    // Free all device memory. Safe to call multiple times.
    void reset();

    // Spin-acquire a free slot (yields between attempts, max 5ms).
    // Throws AscendMemoryPressureException if memory usage exceeds safety threshold.
    // Returns nullptr only on timeout (should not happen in normal operation).
    AscendPooledBuffer* acquire();

    // Mark slot as available. Must be called exactly once per acquire().
    void release(AscendPooledBuffer* buf);

    int  poolSize() const { return static_cast<int>(slots_.size()); }
    bool isSafe()   const;

    static constexpr float kOomGuardRatio = 0.85f;

private:
    MemoryChecker                   memory_checker_;
    std::vector<AscendPooledBuffer> slots_;
    int                             device_id_{0};
    aclrtContext                    ctx_{nullptr};       // bound before aclrtFree in reset()
    bool                            test_mode_{false};  // true → operator new path
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
