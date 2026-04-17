#pragma once

#include "pipeline/IStage.h"
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace infer {

class JoinByFrameStage final : public IStage {
public:
    explicit JoinByFrameStage(std::string id);

    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    struct JoinState {
        std::optional<InferResult> infer;
        std::optional<ArchiveInfo> archive;
        std::chrono::steady_clock::time_point created_at{};
        bool initialized{false};
    };
    std::string id_;
    std::mutex mu_;
    std::unordered_map<std::string, JoinState> pending_;
};

} // namespace infer
