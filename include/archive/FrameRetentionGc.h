#pragma once

#include "common/Config.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>

namespace infer {

class FrameRetentionGc {
public:
    explicit FrameRetentionGc(FrameArchiveConfig cfg);
    ~FrameRetentionGc();

    FrameRetentionGc(const FrameRetentionGc&) = delete;
    FrameRetentionGc& operator=(const FrameRetentionGc&) = delete;

    void start();
    void stop();

    // Public for tests: inject wall-clock time, return removed entry count.
    std::uintmax_t runOnce(std::time_t now);

private:
    void loop();

    FrameArchiveConfig      cfg_;
    std::atomic<bool>       stop_{false};
    std::mutex              mu_;
    std::condition_variable cv_;
    std::thread             worker_;
    bool                    started_{false};
};

} // namespace infer
