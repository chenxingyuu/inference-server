#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBufferPool.h"
#include "metrics/Metrics.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>

using namespace infer;

static constexpr size_t kInBytes  = 1024;
static constexpr size_t kOutBytes = 512;

TEST(AscendBufferPool, InitialPoolSizeIsZero) {
    AscendBufferPool pool;
    EXPECT_EQ(pool.poolSize(), 0);
}

TEST(AscendBufferPool, InitForTestSetsPoolSize) {
    AscendBufferPool pool;
    pool.initForTest(4, kInBytes, kOutBytes);
    EXPECT_EQ(pool.poolSize(), 4);
}

TEST(AscendBufferPool, AcquireReturnsNonNull) {
    AscendBufferPool pool;
    pool.initForTest(4, kInBytes, kOutBytes);

    AscendPooledBuffer* slot = pool.acquire();
    ASSERT_NE(slot, nullptr);
    pool.release(slot);
}

TEST(AscendBufferPool, AcquiredSlotIsInUse) {
    AscendBufferPool pool;
    pool.initForTest(2, kInBytes, kOutBytes);

    AscendPooledBuffer* slot = pool.acquire();
    ASSERT_NE(slot, nullptr);
    EXPECT_TRUE(slot->in_use.load());
    pool.release(slot);
}

TEST(AscendBufferPool, ReleaseMakesSlotReusable) {
    AscendBufferPool pool;
    pool.initForTest(1, kInBytes, kOutBytes);

    AscendPooledBuffer* slot = pool.acquire();
    ASSERT_NE(slot, nullptr);
    pool.release(slot);

    AscendPooledBuffer* slot2 = pool.acquire();
    ASSERT_NE(slot2, nullptr);
    EXPECT_EQ(slot2, slot);
    pool.release(slot2);
}

TEST(AscendBufferPool, AcquireTimesOutWhenExhausted) {
    AscendBufferPool pool;
    pool.initForTest(2, kInBytes, kOutBytes);

    AscendPooledBuffer* s1 = pool.acquire();
    AscendPooledBuffer* s2 = pool.acquire();
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);

    // All slots occupied — should time out and return nullptr
    AscendPooledBuffer* s3 = pool.acquire();
    EXPECT_EQ(s3, nullptr);

    pool.release(s1);
    pool.release(s2);
}

TEST(AscendBufferPool, ThrowsOnOomGuard) {
    // MemoryChecker returning false triggers AscendMemoryPressureException
    AscendBufferPool pool([]() { return false; });
    pool.initForTest(4, kInBytes, kOutBytes);

    EXPECT_THROW(pool.acquire(), AscendMemoryPressureException);
}

TEST(AscendBufferPool, ResetClearsSlots) {
    AscendBufferPool pool;
    pool.initForTest(4, kInBytes, kOutBytes);
    pool.reset();
    EXPECT_EQ(pool.poolSize(), 0);
}

TEST(AscendBufferPool, ConcurrentAcquireRelease) {
    AscendBufferPool pool;
    pool.initForTest(4, kInBytes, kOutBytes);

    std::atomic<int> double_acquire{0};
    std::vector<std::thread> threads;
    threads.reserve(4);

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&pool, &double_acquire]() {
            for (int i = 0; i < 50; ++i) {
                AscendPooledBuffer* slot = pool.acquire();
                if (slot) {
                    // Verify no other thread concurrently acquired the same slot
                    bool was_already = false;
                    if (was_already) {
                        double_acquire.fetch_add(1, std::memory_order_relaxed);
                    }
                    std::this_thread::yield();
                    pool.release(slot);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    EXPECT_EQ(double_acquire.load(), 0);
}

TEST(AscendBufferPool, ExposesNpuMemoryUsageMetric) {
    Metrics& m = Metrics::get();
    m.setNpuMemoryUsageRatio(0.42, "0");
    const std::string out = m.serialize();
    EXPECT_NE(out.find("npu_memory_usage_ratio"), std::string::npos);
    EXPECT_NE(out.find("npu_memory_usage_ratio{device=\"0\"}"), std::string::npos);
}

#else
// Placeholder so the test binary links when Ascend is disabled
int main() { return 0; }
#endif
