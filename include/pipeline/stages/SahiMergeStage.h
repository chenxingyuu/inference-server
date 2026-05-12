#pragma once

#include "pipeline/IStage.h"

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace infer {

struct SahiMergeConfig {
    float merge_iou{0.55f};
    int stale_timeout_ms{2000};
};

class SahiMergeStage final : public IStage {
public:
    SahiMergeStage(std::string id, SahiMergeConfig cfg);
    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    struct PendingState {
        std::chrono::steady_clock::time_point created_at{};
        std::shared_ptr<Frame> parent_frame;
        std::optional<InferResult> template_result;
        std::vector<Detection> detections;
        int expected_tiles{0};
        int received_tiles{0};
        uint64_t parent_frame_seq{0};
        std::string stream_id;
        std::string event_id;
    };

    using PendingKey = std::pair<std::string, uint64_t>;

    static float iou(const BBox& a, const BBox& b);
    std::vector<Detection> runNms(const std::vector<Detection>& detections) const;
    void sweepStale();

    std::string id_;
    SahiMergeConfig cfg_;
    std::mutex mu_;
    std::map<PendingKey, PendingState> pending_;
};

} // namespace infer
