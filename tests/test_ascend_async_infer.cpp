#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBackend.h"
#include "infer/AscendBufferPool.h"
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <atomic>
#include <stdexcept>

using namespace infer;

// ── Helpers ────────────────────────────────────────────────────────────────

// Build a minimal AscendBackend pre-loaded with a test pool (no real NPU).
// The backend is put into a state where infer() can be called via stubs.
static AscendBackend makeTestBackend() {
    AscendBackend backend;
    backend.initPoolForTest(4, /*input_bytes=*/1024, /*output_bytes=*/512);
    return backend;
}

// ── Tests ──────────────────────────────────────────────────────────────────

TEST(AscendAsyncInfer, UsesExecuteAsyncNotSync) {
    std::atomic<int> async_calls{0};
    std::atomic<int> sync_calls{0};

    AscendBackend backend;
    backend.initPoolForTest(4, 1024, 512);
    backend.setExecuteFnForTest(
        [&async_calls, &sync_calls](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            async_calls.fetch_add(1);
            (void)sync_calls; // sync version never called
            return ACL_SUCCESS;
        }
    );
    backend.setSyncStreamFnForTest([](aclrtStream) -> aclError { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest([](void*, size_t, const void*, size_t, aclrtMemcpyKind) -> aclError {
        return ACL_SUCCESS;
    });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
    std::vector<float> output;
    backend.inferWithStubs(batch, output);

    EXPECT_EQ(async_calls.load(), 1);
    EXPECT_EQ(sync_calls.load(), 0);
}

TEST(AscendAsyncInfer, SynchronizesStreamAfterAsync) {
    std::atomic<int> exec_order_exec{0};
    std::atomic<int> exec_order_sync{0};
    std::atomic<int> counter{0};

    AscendBackend backend;
    backend.initPoolForTest(4, 1024, 512);
    backend.setExecuteFnForTest(
        [&exec_order_exec, &counter](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            exec_order_exec.store(counter.fetch_add(1));
            return ACL_SUCCESS;
        }
    );
    backend.setSyncStreamFnForTest([&exec_order_sync, &counter](aclrtStream) -> aclError {
        exec_order_sync.store(counter.fetch_add(1));
        return ACL_SUCCESS;
    });
    backend.setMemcpyFnForTest([](void*, size_t, const void*, size_t, aclrtMemcpyKind) -> aclError {
        return ACL_SUCCESS;
    });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
    std::vector<float> output;
    backend.inferWithStubs(batch, output);

    // ExecuteAsync must have been called before SynchronizeStream
    EXPECT_LT(exec_order_exec.load(), exec_order_sync.load());
}

TEST(AscendAsyncInfer, PoolSlotReleasedAfterInfer) {
    AscendBackend backend;
    backend.initPoolForTest(1, 1024, 512);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; }
    );
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest([](void*, size_t, const void*, size_t, aclrtMemcpyKind) {
        return ACL_SUCCESS;
    });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
    std::vector<float> output;
    backend.inferWithStubs(batch, output);

    // After successful infer, the pool should be able to acquire again
    EXPECT_NO_THROW({
        Batch batch2;
        batch2.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
        std::vector<float> output2;
        backend.inferWithStubs(batch2, output2);
    });
}

TEST(AscendAsyncInfer, PoolSlotReleasedOnError) {
    AscendBackend backend;
    backend.initPoolForTest(1, 1024, 512);
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            return ACL_ERROR_FAILURE;  // simulate NPU error
        }
    );
    backend.setSyncStreamFnForTest([](aclrtStream) { return ACL_SUCCESS; });
    backend.setMemcpyFnForTest([](void*, size_t, const void*, size_t, aclrtMemcpyKind) {
        return ACL_SUCCESS;
    });

    Batch batch;
    batch.frames.push_back(cv::Mat(640, 640, CV_8UC3, cv::Scalar(0)));
    std::vector<float> output;

    EXPECT_THROW(backend.inferWithStubs(batch, output), std::runtime_error);

    // Pool slot must be released even after error (exception safety)
    // Verify by checking a second call can acquire a slot
    backend.setExecuteFnForTest(
        [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) { return ACL_SUCCESS; }
    );
    EXPECT_NO_THROW(backend.inferWithStubs(batch, output));
}

#else
int main() { return 0; }
#endif
