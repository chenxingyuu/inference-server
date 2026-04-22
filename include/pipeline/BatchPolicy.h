#pragma once

#include <algorithm>

namespace infer {

// Adaptive batch-size policy for GPU fault self-healing.
//
// On OOM: halves current_max (min 1) — reduces GPU memory pressure.
// On success: increments streak; restores one tier after kRestoreThreshold successes.
// isExhausted(): true when stuck at current_max=1 with >3 accumulated OOM events.
//
// Header-only pure logic — no CUDA or threading dependency.
struct BatchPolicy {
    int current_max   = 16; // current allowed maximum batch size
    int oom_count     = 0;  // cumulative OOM events
    int success_streak = 0; // consecutive successful inferences

    // Number of consecutive successes required to restore one batch-size tier.
    static constexpr int kRestoreThreshold = 120;

    void onOOM() {
        ++oom_count;
        current_max   = std::max(1, current_max / 2);
        success_streak = 0;
    }

    void onSuccess() {
        if (++success_streak >= kRestoreThreshold) {
            current_max   = std::min(16, current_max * 2);
            success_streak = 0;
        }
    }

    // True when the policy has degraded to single-frame batches and keeps failing.
    bool isExhausted() const {
        return current_max == 1 && oom_count > 3;
    }
};

} // namespace infer
