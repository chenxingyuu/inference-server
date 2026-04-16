#pragma once

#include "common/Config.h"
#include "tracker/ITracker.h"

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>

namespace infer {

class TrackerManager {
public:
    TrackerManager() = default;

    // Applies tracking for one stream/frame in-place on detections.
    void apply(const std::string& stream_id,
               TrackerType tracker_type,
               const ByteTrackConfig& byte_track_cfg,
               uint64_t frame_seq,
               std::vector<Detection>& detections);

private:
    std::unique_ptr<ITracker> createTracker(TrackerType tracker_type,
                                            const ByteTrackConfig& byte_track_cfg) const;
    static bool isSameByteTrackConfig(const ByteTrackConfig& a, const ByteTrackConfig& b);

    std::mutex mu_;
    std::unordered_map<std::string, std::unique_ptr<ITracker>> trackers_;
    std::unordered_map<std::string, ByteTrackConfig> bytetrack_cfgs_;
    std::unordered_set<std::string> deepsort_warned_streams_;
};

} // namespace infer
