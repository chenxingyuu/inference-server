#pragma once

#include "publisher/ControlPublisher.h"
#include <memory>
#include <mutex>

namespace infer {

// Minimal global bus to emit control events from deep runtime components
// (e.g. FFmpegDecoder) without threading a publisher dependency everywhere.
class ControlEventBus {
public:
    static ControlEventBus& get();

    void setPublisher(std::shared_ptr<ControlPublisher> publisher);
    void clearPublisher();

    // Best-effort emit: swallows all exceptions.
    void emit(ControlEventType type,
              const std::string& stream_id,
              StreamState state,
              double engine_ts,
              int attempt,
              int max_attempts,
              const std::string& reason);

private:
    ControlEventBus() = default;

    std::mutex mu_;
    std::shared_ptr<ControlPublisher> publisher_;
};

} // namespace infer

