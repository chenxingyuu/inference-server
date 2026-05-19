#pragma once

#include "pipeline/IStage.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace infer {

struct SahiSchedulerConfig {
    int tile_width{960};
    int tile_height{960};
    float x_overlap_ratio{0.25f};
    float y_overlap_ratio{0.0f};
    int full_interval{5};
    float roi_expand_ratio{1.3f};
    int max_tiles_per_frame{48};
    int min_roi_width{640};
    int min_roi_height{640};
    int roi_max_age_frames{15};
    int fallback_full_min_gap_frames{10};
};

class SahiSchedulerStage final : public IStage {
public:
    SahiSchedulerStage(std::string id, SahiSchedulerConfig cfg);
    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    struct StreamState {
        uint64_t frame_count{0};
        uint64_t next_tile_seq{1};
        uint64_t last_fallback_full_frame{0};
        std::chrono::steady_clock::time_point last_seen{};
    };

    std::vector<int> makeAxisStarts(int full, int tile, float overlap) const;
    std::vector<cv::Rect> makeFullTiles(int full_w, int full_h) const;
    std::vector<cv::Rect> makeRoiTiles(int full_w, int full_h, const std::string& stream_id) const;
    uint64_t nextTileSeqLocked(const std::string& stream_id);
    void sweepStaleStreamsLocked();

    std::string id_;
    SahiSchedulerConfig cfg_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, StreamState> stream_state_;
    std::chrono::steady_clock::time_point last_sweep_{std::chrono::steady_clock::now()};
};

} // namespace infer
