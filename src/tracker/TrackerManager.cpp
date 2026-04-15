#include "tracker/TrackerManager.h"

#include "common/Logger.h"
#include "tracker/ByteTrackTracker.h"

#include <stdexcept>

namespace infer {

std::unique_ptr<ITracker> TrackerManager::createTracker(TrackerType tracker_type) const {
    if (tracker_type == TrackerType::ByteTrack) {
        return std::make_unique<ByteTrackTracker>();
    }
    throw std::runtime_error("Tracker type is not implemented in createTracker");
}

void TrackerManager::apply(const std::string& stream_id,
                           TrackerType tracker_type,
                           uint64_t frame_seq,
                           std::vector<Detection>& detections) {
    std::lock_guard<std::mutex> lk(mu_);

    if (tracker_type == TrackerType::None) {
        trackers_.erase(stream_id);
        return;
    }

    if (tracker_type == TrackerType::DeepSort) {
        trackers_.erase(stream_id);
        if (deepsort_warned_streams_.insert(stream_id).second) {
            LOG_WARN("Tracker for stream {} is set to deepsort, but DeepSORT is not implemented yet", stream_id);
        }
        return;
    }

    auto it = trackers_.find(stream_id);
    if (it == trackers_.end()) {
        auto tracker = createTracker(tracker_type);
        it = trackers_.emplace(stream_id, std::move(tracker)).first;
        LOG_INFO("TrackerManager: created ByteTrack tracker for stream {}", stream_id);
    }
    it->second->update(frame_seq, detections);
}

} // namespace infer
