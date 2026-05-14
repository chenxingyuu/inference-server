// test_ascend_dvpp_zerocopy.cpp
//
// TDD tests for the DVPP → AIPP zero-copy pipeline.
//
// Goal: when a Batch carries AscendBuffer frames (DVPP NV12 in HBM) and the
// model has AIPP enabled, AscendBackend must forward the device pointer
// directly to the NPU — no CPU resize, no H2D memcpy.
//
// All tests use injected stubs; no real NPU hardware is required.

#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBackend.h"
#include "common/Types.h"
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <atomic>
#include <cstdint>

using namespace infer;

// ── Helpers ────────────────────────────────────────────────────────────────

// Sentinel device pointer value used in tests (non-null, clearly fake).
static constexpr uintptr_t kFakeDevPtr = 0xDEAD'C0DE'0000'0001ULL;

// Build a Batch with `n` is_ascend frames at the given resolution.
static Batch makeAscendBatch(int n, int w = 1920, int h = 1080) {
    Batch b;
    b.is_ascend = true;
    // Compute DVPP-aligned dims so AscendBackend uses the correct NV12 stride.
    const int aligned_w = (w + 15) & ~15;
    const int aligned_h = (h +  1) & ~1;
    for (int i = 0; i < n; ++i) {
        AscendBuffer buf;
        // Offset each frame's fake device pointer so we can verify individual
        // D2D copies in the multi-frame path.
        buf.yuv_device    = reinterpret_cast<void*>(kFakeDevPtr + static_cast<uintptr_t>(i));
        buf.width          = w;
        buf.height         = h;
        buf.aligned_width  = aligned_w;
        buf.aligned_height = aligned_h;
        buf.device_id      = 0;
        b.ascend_frames.push_back(std::move(buf));
        StreamMeta m;
        m.stream_id = "cam" + std::to_string(i);
        b.metas.push_back(std::move(m));
    }
    return b;
}

// Build a backend with AIPP enabled, test pool, and no-op ACL stubs.
// dev_alloc and dev_free stubs default to no-ops that record calls.
static AscendBackend makeAippBackend(size_t in_bytes = 1920 * 1080 * 3 / 2) {
    AscendBackend backend;
    backend.initPoolForTest(4, in_bytes, /*output_bytes=*/1024);
    backend.setAippEnabledForTest(true);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; });
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });
    return backend;
}

// ── Tests ──────────────────────────────────────────────────────────────────

// ── 1. Batch struct ────────────────────────────────────────────────────────

TEST(DvppZeroCopy, BatchIsAscendDefaultsFalse) {
    Batch b;
    EXPECT_FALSE(b.is_ascend);
    EXPECT_TRUE(b.ascend_frames.empty());
}

TEST(DvppZeroCopy, BatchAscendFramesCarryDevicePtr) {
    Batch b = makeAscendBatch(2, 1280, 720);
    EXPECT_TRUE(b.is_ascend);
    ASSERT_EQ(static_cast<int>(b.ascend_frames.size()), 2);
    EXPECT_EQ(b.ascend_frames[0].width, 1280);
    EXPECT_EQ(b.ascend_frames[0].height, 720);
    EXPECT_NE(b.ascend_frames[0].yuv_device, nullptr);
    EXPECT_NE(b.ascend_frames[1].yuv_device, nullptr);
    // Each frame has a distinct device pointer
    EXPECT_NE(b.ascend_frames[0].yuv_device, b.ascend_frames[1].yuv_device);
}

// ── 2. Batch=1 zero-copy: no D2D alloc, no D2D memcpy ─────────────────────

TEST(DvppZeroCopy, Batch1_DevAllocNotCalled) {
    AscendBackend backend = makeAippBackend();

    std::atomic<int> alloc_calls{0};
    backend.setDevAllocFnForTest(
        [&alloc_calls](void** ptr, size_t) -> aclError {
            alloc_calls.fetch_add(1);
            *ptr = nullptr;
            return ACL_SUCCESS;
        });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(1);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    // For batch=1 the DVPP ptr is used directly — no temporary allocation.
    EXPECT_EQ(alloc_calls.load(), 0);
}

TEST(DvppZeroCopy, Batch1_D2DMemcpyNotCalled) {
    AscendBackend backend = makeAippBackend();

    std::atomic<int> d2d_calls{0};
    // Override memcpy stub: count D2D calls specifically.
    backend.setMemcpyFnForTest(
        [&d2d_calls](void*, size_t, const void*, size_t,
                     aclrtMemcpyKind kind) -> aclError {
            if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
                d2d_calls.fetch_add(1);
            }
            return ACL_SUCCESS;
        });
    backend.setDevAllocFnForTest([](void** p, size_t) { *p = nullptr; return ACL_SUCCESS; });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(1);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(d2d_calls.load(), 0) << "batch=1 must not D2D-copy; use DVPP ptr directly";
}

TEST(DvppZeroCopy, Batch1_ProducesOutput) {
    AscendBackend backend = makeAippBackend(/*in_bytes=*/1024);
    backend.setDevAllocFnForTest([](void** p, size_t) { *p = nullptr; return ACL_SUCCESS; });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(1);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));
    // Output vector must be resized (size comes from output_bytes / sizeof(float))
    EXPECT_EQ(output.size(), 1024u / sizeof(float));
}

// ── 3. Batch>1 D2D concatenation ───────────────────────────────────────────

TEST(DvppZeroCopy, BatchN_DevAllocCalledOnce) {
    AscendBackend backend = makeAippBackend();

    std::atomic<int> alloc_calls{0};
    backend.setDevAllocFnForTest(
        [&alloc_calls](void** ptr, size_t sz) -> aclError {
            alloc_calls.fetch_add(1);
            // Provide a real heap buffer so D2D "copies" don't crash.
            *ptr = ::operator new(sz);
            return ACL_SUCCESS;
        });
    backend.setDevFreeFnForTest([](void* p) -> aclError {
        ::operator delete(p);
        return ACL_SUCCESS;
    });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(3);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(alloc_calls.load(), 1) << "exactly one temporary device buffer allocation";
}

TEST(DvppZeroCopy, BatchN_D2DCopiedOncePerFrame) {
    constexpr int kBatchSize = 4;
    AscendBackend backend = makeAippBackend();

    // Provide a real backing buffer for the tmp allocation
    std::vector<char> backing(4 * 1920 * 1080 * 3 / 2);
    backend.setDevAllocFnForTest([&backing](void** p, size_t) -> aclError {
        *p = backing.data();
        return ACL_SUCCESS;
    });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    std::atomic<int> d2d_calls{0};
    backend.setMemcpyFnForTest(
        [&d2d_calls](void*, size_t, const void*, size_t,
                     aclrtMemcpyKind kind) -> aclError {
            if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
                d2d_calls.fetch_add(1);
            }
            return ACL_SUCCESS;
        });

    Batch batch = makeAscendBatch(kBatchSize);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(d2d_calls.load(), kBatchSize)
        << "one D2D copy per frame when concatenating into tmp buffer";
}

TEST(DvppZeroCopy, BatchN_TmpBufferFreedAfterInfer) {
    AscendBackend backend = makeAippBackend();

    std::atomic<int> free_calls{0};
    std::vector<char> backing(4 * 1920 * 1080 * 3 / 2);
    backend.setDevAllocFnForTest([&backing](void** p, size_t) -> aclError {
        *p = backing.data();
        return ACL_SUCCESS;
    });
    backend.setDevFreeFnForTest([&free_calls](void*) -> aclError {
        free_calls.fetch_add(1);
        return ACL_SUCCESS;
    });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(2);
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(free_calls.load(), 1) << "tmp buffer must be freed after infer";
}

TEST(DvppZeroCopy, BatchN_TmpBufferFreedEvenOnExecError) {
    // Verify exception safety: tmp buffer is freed even when Execute fails.
    AscendBackend backend = makeAippBackend();

    std::atomic<int> free_calls{0};
    std::vector<char> backing(4 * 1920 * 1080 * 3 / 2);
    backend.setDevAllocFnForTest([&backing](void** p, size_t) -> aclError {
        *p = backing.data();
        return ACL_SUCCESS;
    });
    backend.setDevFreeFnForTest([&free_calls](void*) -> aclError {
        free_calls.fetch_add(1);
        return ACL_SUCCESS;
    });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            return ACL_ERROR_FAILURE;
        });

    Batch batch = makeAscendBatch(2);
    std::vector<float> output;
    EXPECT_THROW(backend.inferWithStubs(batch, output), std::runtime_error);

    EXPECT_EQ(free_calls.load(), 1) << "tmp buffer must be freed even on error";
}

// ── 4. CPU fallback paths (backward compatibility) ─────────────────────────

TEST(DvppZeroCopy, CpuFrames_AippEnabled_PackBgrPath) {
    // CPU frames + AIPP → packBgrUint8 path; dev_alloc must NOT be called.
    AscendBackend backend;
    backend.initPoolForTest(4, 640 * 640 * 3, 1024);
    backend.setAippEnabledForTest(true);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; });
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });

    std::atomic<int> alloc_calls{0};
    backend.setDevAllocFnForTest([&alloc_calls](void** p, size_t) -> aclError {
        alloc_calls.fetch_add(1);
        *p = nullptr;
        return ACL_SUCCESS;
    });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(alloc_calls.load(), 0) << "CPU frame path must not touch device allocator";
}

TEST(DvppZeroCopy, CpuFrames_AippDisabled_PreprocessCpuPath) {
    AscendBackend backend;
    backend.initPoolForTest(4, 640 * 640 * 3 * sizeof(float), 1024);
    backend.setAippEnabledForTest(false);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; });
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });

    std::atomic<int> alloc_calls{0};
    backend.setDevAllocFnForTest([&alloc_calls](void** p, size_t) -> aclError {
        alloc_calls.fetch_add(1);
        *p = nullptr;
        return ACL_SUCCESS;
    });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(128)));
    std::vector<float> output;
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));

    EXPECT_EQ(alloc_calls.load(), 0);
}

TEST(DvppZeroCopy, AscendBatch_AippDisabled_FallsBackToCpu) {
    // Even if is_ascend=true, without AIPP the CPU path must be used.
    // The backend falls back to packBgrUint8 (using frame.image if available)
    // or skips preprocess if no CPU frames. Verify no crash.
    AscendBackend backend;
    backend.initPoolForTest(4, 1024, 1024);
    backend.setAippEnabledForTest(false);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; });
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest(
        [](void*, size_t, const void*, size_t, aclrtMemcpyKind) { return ACL_SUCCESS; });
    backend.setDevAllocFnForTest([](void** p, size_t) { *p = nullptr; return ACL_SUCCESS; });
    backend.setDevFreeFnForTest([](void*) { return ACL_SUCCESS; });

    Batch batch = makeAscendBatch(1);
    std::vector<float> output;
    // Without AIPP, the ascend path is not taken; batch has no CPU frames so
    // inferWithStubs returns early without crashing.
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));
}

#else
int main() { return 0; }
#endif
