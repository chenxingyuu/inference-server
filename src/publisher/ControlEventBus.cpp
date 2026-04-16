#include "publisher/ControlEventBus.h"
#include "common/Logger.h"

namespace infer {

ControlEventBus& ControlEventBus::get() {
    static ControlEventBus instance;
    return instance;
}

void ControlEventBus::setPublisher(std::shared_ptr<ControlPublisher> publisher) {
    std::lock_guard<std::mutex> lock(mu_);
    publisher_ = std::move(publisher);
}

void ControlEventBus::clearPublisher() {
    std::lock_guard<std::mutex> lock(mu_);
    publisher_.reset();
}

void ControlEventBus::emit(ControlEventType type,
                           const std::string& stream_id,
                           StreamState state,
                           double engine_ts,
                           int attempt,
                           int max_attempts,
                           const std::string& reason) {
    std::shared_ptr<ControlPublisher> p;
    {
        std::lock_guard<std::mutex> lock(mu_);
        p = publisher_;
    }
    if (!p) return;
    try {
        p->publishStreamEvent(type, stream_id, state, engine_ts, attempt, max_attempts, reason);
    } catch (const std::exception& e) {
        LOG_WARN("ControlEventBus: emit failed: {}", e.what());
    } catch (...) {
        LOG_WARN("ControlEventBus: emit failed: unknown");
    }
}

} // namespace infer

