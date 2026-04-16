#pragma once

#include <cstdint>

namespace infer {

struct ReconnectPolicy {
    // 0 means never stop reconnecting.
    uint32_t max_reconnect_attempts{0};
};

// Returns true if the decoder should stop reconnecting (terminal failure).
bool shouldStopReconnect(const ReconnectPolicy& policy, uint32_t consecutive_failures);

} // namespace infer

