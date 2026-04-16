#include "tracker/TrackerManager.h"

#include "common/Logger.h"
#include "tracker/ByteTrackTracker.h"

#include <cmath>
#include <stdexcept>

namespace infer {

std::unique_ptr<ITracker> TrackerManager::createTracker(TrackerType tracker_type,
                                                        const ByteTrackConfig& byte_track_cfg) const {
    if (tracker_type == TrackerType::ByteTrack) {
        return std::make_unique<ByteTrackTracker>(byte_track_cfg);
    }
    throw std::runtime_error("Tracker type is not implemented in createTracker");
}

bool TrackerManager::isSameByteTrackConfig(const ByteTrackConfig& a, const ByteTrackConfig& b) {
    constexpr float kEps = 1e-6f;
    return std::fabs(a.high_det_thresh - b.high_det_thresh) < kEps
        && std::fabs(a.low_det_thresh - b.low_det_thresh) < kEps
        && std::fabs(a.match_iou_thresh - b.match_iou_thresh) < kEps
        && a.min_hits_to_confirm == b.min_hits_to_confirm
        && a.max_lost_frames == b.max_lost_frames;
}

void TrackerManager::apply(const std::string& stream_id,
                           TrackerType tracker_type,
                           const ByteTrackConfig& byte_track_cfg,
                           uint64_t frame_seq,
                           std::vector<Detection>& detections) {
    std::lock_guard<std::mutex> lk(mu_);

    if (tracker_type == TrackerType::None) {
        trackers_.erase(stream_id);
        bytetrack_cfgs_.erase(stream_id);
        return;
    }

    if (tracker_type == TrackerType::DeepSort) {
        trackers_.erase(stream_id);
        bytetrack_cfgs_.erase(stream_id);
        if (deepsort_warned_streams_.insert(stream_id).second) {
            LOG_WARN("Tracker for stream {} is set to deepsort, but DeepSORT is not implemented yet", stream_id);
        }
        return;
    }

    auto it = trackers_.find(stream_id);
    if (it == trackers_.end()) {
        auto tracker = createTracker(tracker_type, byte_track_cfg);
        it = trackers_.emplace(stream_id, std::move(tracker)).first;
        bytetrack_cfgs_[stream_id] = byte_track_cfg;
        LOG_INFO("TrackerManager: created ByteTrack tracker for stream {}", stream_id);
    } else if (!isSameByteTrackConfig(bytetrack_cfgs_[stream_id], byte_track_cfg)) {
        auto tracker = createTracker(tracker_type, byte_track_cfg);
        it->second = std::move(tracker);
        bytetrack_cfgs_[stream_id] = byte_track_cfg;
        LOG_INFO("TrackerManager: re-created ByteTrack tracker for stream {} due to config change", stream_id);
    }
    it->second->update(frame_seq, detections);
}

} // namespace infer
