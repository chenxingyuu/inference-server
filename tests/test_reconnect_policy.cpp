#include <gtest/gtest.h>
#include "stream/ReconnectPolicy.h"

using namespace infer;

TEST(ReconnectPolicy, StopAfterMaxReconnectAttempts) {
    ReconnectPolicy cfg;
    cfg.max_reconnect_attempts = 5;
    EXPECT_FALSE(shouldStopReconnect(cfg, /*consecutive_failures=*/4));
    EXPECT_TRUE(shouldStopReconnect(cfg, /*consecutive_failures=*/5));
    EXPECT_TRUE(shouldStopReconnect(cfg, /*consecutive_failures=*/6));
}

TEST(ReconnectPolicy, ZeroMaxMeansNeverStop) {
    ReconnectPolicy cfg;
    cfg.max_reconnect_attempts = 0;
    for (uint32_t f = 0; f < 100; ++f) {
        EXPECT_FALSE(shouldStopReconnect(cfg, f));
    }
}

