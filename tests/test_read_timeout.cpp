#include <gtest/gtest.h>
#include "stream/InterruptCtx.h"
#include "common/Config.h"
#include <thread>
#include <chrono>

using infer::InterruptCtx;

// ─── InterruptCtx behaviour ───────────────────────────────────────────────────

TEST(InterruptCtx, DisarmedReturnsZero) {
    InterruptCtx ctx;
    EXPECT_EQ(InterruptCtx::check(&ctx), 0);
}

TEST(InterruptCtx, ArmedBeforeDeadlineReturnsZero) {
    InterruptCtx ctx;
    ctx.arm(10'000'000LL); // 10 seconds — well in the future
    EXPECT_EQ(InterruptCtx::check(&ctx), 0);
}

TEST(InterruptCtx, ArmedAfterDeadlineReturnsOne) {
    InterruptCtx ctx;
    ctx.arm(1'000LL); // 1 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(InterruptCtx::check(&ctx), 1);
}

TEST(InterruptCtx, DisarmAfterArmReturnsZero) {
    InterruptCtx ctx;
    ctx.arm(1'000LL);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ctx.disarm();
    EXPECT_EQ(InterruptCtx::check(&ctx), 0);
}

// ─── StreamConfig / PipelineSourceConfig default values ──────────────────────

TEST(ReadTimeoutConfig, StreamConfigDefaults) {
    infer::StreamConfig cfg;
    EXPECT_EQ(cfg.open_timeout_ms, 10000);
    EXPECT_EQ(cfg.read_timeout_ms, 5000);
}

TEST(ReadTimeoutConfig, PipelineSourceConfigDefaults) {
    infer::PipelineSourceConfig src;
    EXPECT_EQ(src.open_timeout_ms, 10000);
    EXPECT_EQ(src.read_timeout_ms, 5000);
}
