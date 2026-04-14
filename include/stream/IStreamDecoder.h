#pragma once

#include "common/Types.h"
#include "common/Config.h"
#include <functional>

namespace infer {

// Callback invoked with each decoded frame (called from decoder thread)
using FrameCallback = std::function<void(Frame)>;

class IStreamDecoder {
public:
    virtual ~IStreamDecoder() = default;

    // Start background decode loop; returns immediately
    virtual void start(const StreamConfig& cfg, FrameCallback cb) = 0;

    // Signal the decode loop to stop; blocks until thread exits
    virtual void stop() = 0;

    // True if the decoder thread is running
    virtual bool running() const = 0;

    virtual const std::string& streamId() const = 0;
};

} // namespace infer
