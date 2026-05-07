#include "stream/StopAwareSleep.h"

#include <thread>

namespace infer {

bool waitForOrStop(const std::atomic<bool>& stop_flag, std::chrono::milliseconds delay) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + delay;
    while (!stop_flag.load()) {
        const auto now = steady_clock::now();
        if (now >= deadline) return false;
        const auto remaining = duration_cast<milliseconds>(deadline - now);
        std::this_thread::sleep_for(std::min(milliseconds(20), remaining));
    }
    return true;
}

} // namespace infer
