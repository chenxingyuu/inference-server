#include <gtest/gtest.h>

#include "stream/StopAwareSleep.h"

#include <atomic>
#include <chrono>
#include <thread>

TEST(StopAwareSleep, ReturnsTrueSoonWhenStopFlagSet) {
    std::atomic<bool> stop{false};
    std::thread stopper([&stop] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true);
    });

    const auto start = std::chrono::steady_clock::now();
    const bool interrupted = infer::waitForOrStop(stop, std::chrono::milliseconds(500));
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    stopper.join();
    EXPECT_TRUE(interrupted);
    EXPECT_LT(elapsed.count(), 200);
}

TEST(StopAwareSleep, ReturnsFalseAfterTimeoutWhenNotStopped) {
    std::atomic<bool> stop{false};
    const auto start = std::chrono::steady_clock::now();
    const bool interrupted = infer::waitForOrStop(stop, std::chrono::milliseconds(60));
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    EXPECT_FALSE(interrupted);
    EXPECT_GE(elapsed.count(), 50);
}
