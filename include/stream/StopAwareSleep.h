#pragma once

#include <atomic>
#include <chrono>

namespace infer {

// Waits for at most delay; returns true if stop_flag becomes true before timeout.
bool waitForOrStop(const std::atomic<bool>& stop_flag, std::chrono::milliseconds delay);

} // namespace infer
