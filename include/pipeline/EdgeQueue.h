#pragma once

#include "common/Config.h"
#include "pipeline/Event.h"
#include <condition_variable>
#include <deque>
#include <mutex>

namespace infer {

class EdgeQueue {
public:
    explicit EdgeQueue(const EdgeConfig& cfg) : cfg_(cfg) {}

    bool push(const EventEnvelope& event);
    bool pop(EventEnvelope& event, int wait_ms);
    void stop();
    void reopen();
    std::size_t size() const;

private:
    EdgeConfig cfg_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<EventEnvelope> queue_;
    bool stopped_{false};
};

} // namespace infer
