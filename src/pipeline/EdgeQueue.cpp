#include "pipeline/EdgeQueue.h"

namespace infer {

bool EdgeQueue::push(const EventEnvelope& event) {
    std::unique_lock<std::mutex> lock(mu_);
    if (stopped_) return false;

    if (queue_.size() >= static_cast<std::size_t>(cfg_.capacity)) {
        if (cfg_.drop_policy == EdgeDropPolicy::DropNewest) return false;
        if (cfg_.drop_policy == EdgeDropPolicy::DropOldest) {
            queue_.pop_front();
        } else {
            const bool has_space = cv_.wait_for(
                lock, std::chrono::milliseconds(100),
                [this] { return stopped_ || queue_.size() < static_cast<std::size_t>(cfg_.capacity); });
            if (!has_space || stopped_) return false;
        }
    }
    queue_.push_back(event);
    cv_.notify_all();
    return true;
}

bool EdgeQueue::pop(EventEnvelope& event, int wait_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this] { return stopped_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    event = queue_.front();
    queue_.pop_front();
    cv_.notify_all();
    return true;
}

void EdgeQueue::stop() {
    std::lock_guard<std::mutex> lock(mu_);
    stopped_ = true;
    cv_.notify_all();
}

std::size_t EdgeQueue::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

} // namespace infer
