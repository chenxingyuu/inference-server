#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace infer {

// Watchdog for FFmpeg blocking calls (avformat_open_input, av_read_frame, etc.).
//
// Usage:
//   ctx.arm(timeout_us);          // set deadline = now + timeout
//   int ret = av_read_frame(...); // FFmpeg polls check() internally
//   ctx.disarm();                 // clear deadline after call returns
//
// Attach to AVFormatContext before any blocking call:
//   fmt_ctx->interrupt_callback = {InterruptCtx::check, &ctx};
struct InterruptCtx {
    std::atomic<int64_t> deadline_us{0};

    // FFmpeg interrupt callback. Returns 1 when the deadline has passed.
    static int check(void* opaque) {
        auto* ctx = static_cast<InterruptCtx*>(opaque);
        int64_t d = ctx->deadline_us.load(std::memory_order_relaxed);
        if (d == 0) return 0;
        using namespace std::chrono;
        auto now_us = duration_cast<microseconds>(
            steady_clock::now().time_since_epoch()).count();
        return (now_us > d) ? 1 : 0;
    }

    void arm(int64_t timeout_us) {
        using namespace std::chrono;
        auto now_us = duration_cast<microseconds>(
            steady_clock::now().time_since_epoch()).count();
        deadline_us.store(now_us + timeout_us, std::memory_order_relaxed);
    }

    void disarm() {
        deadline_us.store(0, std::memory_order_relaxed);
    }
};

} // namespace infer
