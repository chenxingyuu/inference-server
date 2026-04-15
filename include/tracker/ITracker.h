#pragma once

#include "common/Types.h"
#include <vector>
#include <cstdint>

namespace infer {

class ITracker {
public:
    virtual ~ITracker() = default;

    // Updates track state for a single frame and writes track ids into detections.
    virtual void update(uint64_t frame_seq, std::vector<Detection>& detections) = 0;
};

} // namespace infer
