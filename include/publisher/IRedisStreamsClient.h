#pragma once
#include <string>
#include <vector>
#include <utility>

namespace infer {

// Thin seam over redis-plus-plus XADD for testability.
// The production implementation wraps sw::redis::Redis.
struct IRedisStreamsClient {
    virtual ~IRedisStreamsClient() = default;
    virtual void xadd(const std::string& key,
                      const std::vector<std::pair<std::string, std::string>>& fields,
                      long long maxlen) = 0;
};

} // namespace infer
