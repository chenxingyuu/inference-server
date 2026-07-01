#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include "infer/IInferBackend.h"
#include "infer/AscendBufferPool.h"
#include <acl/acl.h>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>
#include <array>
#include <mutex>
#include <chrono>

namespace infer {

// Supported batch sizes for pre-compiled .om files
static constexpr std::array<int, 4> kAscendBatchSizes = {1, 4, 8, 16};

class AscendBackend : public IInferBackend {
public:
    AscendBackend();
    ~AscendBackend() override;

    void loadModel(const ModelConfig& cfg) override;
    void infer(const Batch& input, std::vector<float>& output) override;
    void unloadModel() override;

    int        maxBatchSize() const override { return max_batch_size_; }
    DeviceType deviceType()   const override { return DeviceType::Ascend; }
    bool       isLoaded()     const override { return loaded_; }

    // ── Test hooks (no real NPU required) ────────────────────────────────

    // Initialise the buffer pool using operator new instead of aclrtMalloc.
    void initPoolForTest(int pool_size, size_t input_bytes, size_t output_bytes);

    // Stub for aclmdlExecuteAsync: inject a callable to verify async path.
    using AclExecuteAsyncFn = std::function<
        aclError(uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream)>;
    void setExecuteFnForTest(AclExecuteAsyncFn fn) { execute_async_fn_ = std::move(fn); }

    // Stub for aclrtSynchronizeStream.
    using AclSyncStreamFn = std::function<aclError(aclrtStream)>;
    void setSyncStreamFnForTest(AclSyncStreamFn fn) { sync_stream_fn_ = std::move(fn); }

    // Optional timed sync hook used to enforce infer timeout.
    using AclSyncStreamWithTimeoutFn = std::function<aclError(aclrtStream, uint32_t)>;
    void setSyncStreamWithTimeoutFnForTest(AclSyncStreamWithTimeoutFn fn) {
        sync_stream_with_timeout_fn_ = std::move(fn);
    }

    // Stub for aclrtMemcpy (H2D + D2H).
    using AclMemcpyFn = std::function<
        aclError(void*, size_t, const void*, size_t, aclrtMemcpyKind)>;
    void setMemcpyFnForTest(AclMemcpyFn fn) { memcpy_fn_ = std::move(fn); }

    // Stub for aclrtMemcpyAsync (mainly D2H async copy).
    using AclMemcpyAsyncFn = std::function<
        aclError(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream)>;
    void setMemcpyAsyncFnForTest(AclMemcpyAsyncFn fn) { memcpy_async_fn_ = std::move(fn); }

    // Stub for AIPP detection.
    using AippDetectFn = std::function<bool(uint32_t)>;
    void setAippDetectFnForTest(AippDetectFn fn) { aipp_detect_fn_ = std::move(fn); }

    // Directly set AIPP flag (for tests that bypass loadModel()).
    void setAippEnabledForTest(bool enabled) { aipp_enabled_ = enabled; }

    // Return the input_bytes_ used to initialise the pool (for test assertions).
    size_t poolInputBytesForTest() const { return input_bytes_; }
    void setInferTimeoutMsForTest(uint32_t timeout_ms) { infer_timeout_ms_ = timeout_ms; }
    void setModelShapeForTest(int h, int w, int num_classes) {
        input_h_ = h;
        input_w_ = w;
        num_classes_ = num_classes;
        output_bytes_ = expectedOutputBytesForBatchForTest(max_batch_size_);
    }
    size_t expectedOutputBytesForBatchForTest(int batch_size) const;

    // Stub for device memory allocation (aclrtMalloc) — zero-copy multi-frame path.
    using AclDevAllocFn = std::function<aclError(void**, size_t)>;
    void setDevAllocFnForTest(AclDevAllocFn fn) { dev_alloc_fn_ = std::move(fn); }

    // Stub for device memory free (aclrtFree) — zero-copy multi-frame path.
    using AclDevFreeFn = std::function<aclError(void*)>;
    void setDevFreeFnForTest(AclDevFreeFn fn) { dev_free_fn_ = std::move(fn); }

    // Run infer() using injected stubs (skips real ACL dataset / model calls).
    // Supports both CPU-frame batches and is_ascend batches (DVPP zero-copy path).
    void inferWithStubs(const Batch& input, std::vector<float>& output);

private:
    // Select the closest .om for the requested batch size (rounds down)
    uint32_t selectModel(int batch_size) const;

    void preprocessCPU(const Batch& input, float* dst,
                       int batch_size, int h, int w);

    void packBgrUint8(const Batch& input, uint8_t* dst,
                      int batch_size, int h, int w);

    bool detectAipp(uint32_t model_id) const;

    int  max_batch_size_{16};
    int  input_h_{640};
    int  input_w_{640};
    int  num_classes_{80};
    bool loaded_{false};
    int  device_id_{0};

    // batch_size → model_id and compiled tensor byte sizes from aclmdlDesc
    std::map<int, uint32_t> model_map_;
    std::map<int, size_t>   model_input_bytes_;
    std::map<int, size_t>   model_output_bytes_;

    aclrtContext ctx_{nullptr};
    aclrtStream  stream_{nullptr};
    std::mutex   infer_mu_;

    // Pre-allocated NPU buffer pool (replaces per-inference aclrtMalloc)
    AscendBufferPool buffer_pool_;
    size_t input_bytes_{0};
    size_t output_bytes_{0};

    bool aipp_enabled_{false};

    // ── Stub function pointers (nullptr → real ACL calls) ─────────────────
    AclExecuteAsyncFn execute_async_fn_{};
    AclSyncStreamFn   sync_stream_fn_{};
    AclSyncStreamWithTimeoutFn sync_stream_with_timeout_fn_{};
    AclMemcpyFn       memcpy_fn_{};
    AclMemcpyAsyncFn  memcpy_async_fn_{};
    AippDetectFn      aipp_detect_fn_{};
    AclDevAllocFn     dev_alloc_fn_{};  // zero-copy: aclrtMalloc for tmp D2D buffer
    AclDevFreeFn      dev_free_fn_{};   // zero-copy: aclrtFree for tmp D2D buffer
    uint32_t          infer_timeout_ms_{5000};
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
