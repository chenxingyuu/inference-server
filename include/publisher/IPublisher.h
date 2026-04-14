#pragma once

#include "common/Types.h"

namespace infer {

class IPublisher {
public:
    virtual ~IPublisher() = default;

    // Asynchronously publish one inference result.
    // Must be thread-safe (called from multiple InferWorker threads).
    virtual void publish(InferResult result) = 0;

    // Flush pending messages and wait for broker acknowledgement (best-effort)
    virtual void flush() = 0;
};

} // namespace infer
