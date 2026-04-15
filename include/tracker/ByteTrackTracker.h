#pragma once

#include "tracker/ITracker.h"
#include <vector>

namespace infer {

class ByteTrackTracker final : public ITracker {
public:
    ByteTrackTracker() = default;
    void update(uint64_t frame_seq, std::vector<Detection>& detections) override;

private:
    struct TrackState {
        int64_t id{0};
        BBox bbox{};
        int age{0};
        int hits{0};
        int lost{0};
        bool confirmed{false};
    };

    static float iou(const BBox& a, const BBox& b);

    // Matches tracks to detection indexes (greedy IoU matching).
    static std::vector<std::pair<int, int>> greedyMatch(
        const std::vector<TrackState*>& tracks,
        const std::vector<int>& det_indices,
        const std::vector<Detection>& detections,
        float iou_thresh);

    int64_t next_id_{1};
    std::vector<TrackState> tracks_;
};

} // namespace infer
