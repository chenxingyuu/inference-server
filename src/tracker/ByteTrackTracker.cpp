#include "tracker/ByteTrackTracker.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace infer {

namespace {
constexpr float kHighDetThresh = 0.5f;
constexpr float kLowDetThresh = 0.1f;
constexpr float kMatchIouThresh = 0.3f;
constexpr int kMinHitsToConfirm = 2;
constexpr int kMaxLostFrames = 30;
} // namespace

float ByteTrackTracker::iou(const BBox& a, const BBox& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float uni = std::max(1e-6f, a.area() + b.area() - inter);
    return inter / uni;
}

std::vector<std::pair<int, int>> ByteTrackTracker::greedyMatch(
    const std::vector<TrackState*>& tracks,
    const std::vector<int>& det_indices,
    const std::vector<Detection>& detections,
    float iou_thresh) {
    std::vector<std::pair<int, int>> matches;
    if (tracks.empty() || det_indices.empty()) {
        return matches;
    }

    std::unordered_set<int> used_tracks;
    std::unordered_set<int> used_dets;
    while (true) {
        float best_iou = -1.0f;
        int best_track_idx = -1;
        int best_det_idx = -1;
        for (int ti = 0; ti < static_cast<int>(tracks.size()); ++ti) {
            if (used_tracks.count(ti) != 0) {
                continue;
            }
            for (int di = 0; di < static_cast<int>(det_indices.size()); ++di) {
                if (used_dets.count(di) != 0) {
                    continue;
                }
                const float score = iou(tracks[ti]->bbox, detections[det_indices[di]].bbox);
                if (score > best_iou) {
                    best_iou = score;
                    best_track_idx = ti;
                    best_det_idx = di;
                }
            }
        }
        if (best_track_idx < 0 || best_det_idx < 0 || best_iou < iou_thresh) {
            break;
        }
        used_tracks.insert(best_track_idx);
        used_dets.insert(best_det_idx);
        matches.emplace_back(best_track_idx, best_det_idx);
    }

    return matches;
}

void ByteTrackTracker::update(uint64_t /*frame_seq*/, std::vector<Detection>& detections) {
    std::vector<int> high_det_idxs;
    std::vector<int> low_det_idxs;
    high_det_idxs.reserve(detections.size());
    low_det_idxs.reserve(detections.size());
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (detections[i].confidence >= kHighDetThresh) {
            high_det_idxs.push_back(i);
        } else if (detections[i].confidence >= kLowDetThresh) {
            low_det_idxs.push_back(i);
        }
    }

    std::vector<TrackState*> track_ptrs;
    track_ptrs.reserve(tracks_.size());
    for (auto& t : tracks_) {
        t.age += 1;
        t.lost += 1;
        track_ptrs.push_back(&t);
    }

    const auto high_matches = greedyMatch(track_ptrs, high_det_idxs, detections, kMatchIouThresh);
    std::unordered_set<int> matched_track_ptr_idxs;
    std::unordered_set<int> matched_high_det_list_idxs;
    for (const auto& [ti, di] : high_matches) {
        auto& track = *track_ptrs[ti];
        auto& det = detections[high_det_idxs[di]];
        track.bbox = det.bbox;
        track.hits += 1;
        track.lost = 0;
        if (!track.confirmed && track.hits >= kMinHitsToConfirm) {
            track.confirmed = true;
        }
        det.track_id = track.id;
        matched_track_ptr_idxs.insert(ti);
        matched_high_det_list_idxs.insert(di);
    }

    std::vector<TrackState*> unmatched_tracks;
    unmatched_tracks.reserve(track_ptrs.size());
    for (int i = 0; i < static_cast<int>(track_ptrs.size()); ++i) {
        if (matched_track_ptr_idxs.count(i) == 0) {
            unmatched_tracks.push_back(track_ptrs[i]);
        }
    }
    const auto low_matches = greedyMatch(unmatched_tracks, low_det_idxs, detections, kMatchIouThresh);
    for (const auto& [ti, di] : low_matches) {
        auto& track = *unmatched_tracks[ti];
        auto& det = detections[low_det_idxs[di]];
        track.bbox = det.bbox;
        track.hits += 1;
        track.lost = 0;
        if (!track.confirmed && track.hits >= kMinHitsToConfirm) {
            track.confirmed = true;
        }
        det.track_id = track.id;
    }

    for (int i = 0; i < static_cast<int>(high_det_idxs.size()); ++i) {
        if (matched_high_det_list_idxs.count(i) != 0) {
            continue;
        }
        TrackState t;
        t.id = next_id_++;
        t.bbox = detections[high_det_idxs[i]].bbox;
        t.age = 1;
        t.hits = 1;
        t.lost = 0;
        t.confirmed = false;
        detections[high_det_idxs[i]].track_id = t.id;
        tracks_.push_back(t);
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const TrackState& t) { return t.lost > kMaxLostFrames; }),
        tracks_.end());
}

} // namespace infer
