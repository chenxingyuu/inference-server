#pragma once

#include "common/Types.h"

#include <optional>
#include <string>
#include <vector>

namespace infer {

struct SahiRoiSnapshot {
    uint64_t frame_seq{0};
    std::vector<Detection> detections;
};

class SahiRoiRegistry {
public:
    static void update(const std::string& stream_id, uint64_t frame_seq, const std::vector<Detection>& detections);
    static std::optional<SahiRoiSnapshot> get(const std::string& stream_id);
};

} // namespace infer
