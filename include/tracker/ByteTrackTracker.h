#pragma once

#include "common/Config.h"
#include "tracker/ITracker.h"
#include <vector>

namespace infer {

class ByteTrackTracker final : public ITracker {
public:
    explicit ByteTrackTracker(ByteTrackConfig cfg);
    void update(uint64_t frame_seq, std::vector<Detection>& detections) override;

private:
    enum class TrackStatus { Tracked, Lost, Removed };

    struct TrackState {
        int64_t id{0};
        BBox bbox{};
        float vx{0.0f};
        float vy{0.0f};
        int age{0};
        int hits{0};
        int lost{0};
        bool confirmed{false};
        TrackStatus status{TrackStatus::Tracked};
    };

    static float iou(const BBox& a, const BBox& b);

    // Matches tracks to detection indexes (global optimal IoU assignment).
    static std::vector<std::pair<int, int>> linearAssign(
        const std::vector<TrackState*>& tracks,
        const std::vector<int>& det_indices,
        const std::vector<Detection>& detections,
        float iou_thresh);

    int64_t next_id_{1};
    ByteTrackConfig cfg_{};
    std::vector<TrackState> tracks_;
};

} // namespace infer
