#pragma once

#include "pipeline/Event.h"
#include <functional>
#include <string>

namespace infer {

class IStage {
public:
    using EmitFn = std::function<void(const EventEnvelope&)>;
    virtual ~IStage() = default;
    virtual void start() {}
    virtual void stop() {}
    // GraphExecutor calls this as soon as the pipeline begins shutdown (after edges
    // are stopped, before node worker threads are joined). Stages with background
    // reconnect loops (ffplay / ffmpeg sinks) should suppress reopening outputs here.
    virtual void onGraphExecutorDraining() noexcept {}
    virtual void process(const EventEnvelope& input, const EmitFn& emit) = 0;
    virtual bool isSource() const { return false; }
    virtual std::string id() const = 0;
};

} // namespace infer
