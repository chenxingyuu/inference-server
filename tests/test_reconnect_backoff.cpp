#include <gtest/gtest.h>
#include <cstdint>
#include <algorithm>

// Replicate the backoff function from FFmpegDecoder.cpp for unit testing
static int64_t backoffDelayMs(int base_ms, int ceiling_ms, uint32_t failures) {
    int64_t delay = static_cast<int64_t>(base_ms) * (1LL << std::min(failures, 6u));
    delay = std::min(delay, static_cast<int64_t>(ceiling_ms));
    int64_t jitter = (delay / 10);
    if (jitter > 0)
        delay += (static_cast<int64_t>(rand()) % (jitter * 2 + 1)) - jitter;
    return std::max(delay, static_cast<int64_t>(base_ms));
}

TEST(ReconnectBackoff, ZeroFailuresReturnsBase) {
    // With 0 failures: delay = base * 2^0 = base (before jitter)
    // After jitter the result is still >= base
    int64_t d = backoffDelayMs(3000, 60000, 0);
    EXPECT_GE(d, 3000);
    EXPECT_LE(d, 3000 + 300 + 1);  // base + 10% jitter max
}

TEST(ReconnectBackoff, DoublesWithEachFailure) {
    // Deterministic check: strip jitter by using base=1000, ceiling=1000000
    // With fixed seed the jitter is hard to predict, so just check ordering and ceiling.
    srand(42);
    int64_t prev = 0;
    for (uint32_t f = 0; f < 6; ++f) {
        // Use ceiling high enough that it never clamps
        int64_t d = backoffDelayMs(100, 100000, f);
        EXPECT_GT(d, prev) << "delay should grow with failures (f=" << f << ")";
        prev = d;
    }
}

TEST(ReconnectBackoff, NeverExceedsCeiling) {
    for (uint32_t f = 0; f <= 20; ++f) {
        int64_t d = backoffDelayMs(3000, 60000, f);
        EXPECT_LE(d, 60000 + 6000 + 1)  // ceiling + 10% jitter max
            << "delay exceeded ceiling at f=" << f;
    }
}

TEST(ReconnectBackoff, FailuresBeyond6ClampExponent) {
    // 2^6 = 64; base=1000, ceiling=1000000
    // Failures=6 and failures=20 should produce the same pre-jitter value
    srand(0);
    int64_t d6  = backoffDelayMs(1000, 1000000, 6);
    srand(0);
    int64_t d20 = backoffDelayMs(1000, 1000000, 20);
    EXPECT_EQ(d6, d20);
}

TEST(ReconnectBackoff, AlwaysAtLeastBase) {
    for (uint32_t f = 0; f <= 10; ++f) {
        int64_t d = backoffDelayMs(3000, 60000, f);
        EXPECT_GE(d, 3000) << "delay below base at f=" << f;
    }
}
