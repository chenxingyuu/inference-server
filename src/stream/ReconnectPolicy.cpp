#include "stream/ReconnectPolicy.h"

namespace infer {

bool shouldStopReconnect(const ReconnectPolicy& policy, uint32_t consecutive_failures) {
    if (policy.max_reconnect_attempts == 0) return false;
    return consecutive_failures >= policy.max_reconnect_attempts;
}

} // namespace infer

