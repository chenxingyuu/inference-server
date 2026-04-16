#include "tracker/ByteTrackTracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace infer {

ByteTrackTracker::ByteTrackTracker(ByteTrackConfig cfg) : cfg_(cfg) {}

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

namespace {
float centerX(const BBox& b) { return (b.x1 + b.x2) * 0.5f; }
float centerY(const BBox& b) { return (b.y1 + b.y2) * 0.5f; }
} // namespace

std::vector<std::pair<int, int>> ByteTrackTracker::linearAssign(
    const std::vector<TrackState*>& tracks,
    const std::vector<int>& det_indices,
    const std::vector<Detection>& detections,
    float iou_thresh) {
    std::vector<std::pair<int, int>> matches;
    if (tracks.empty() || det_indices.empty()) {
        return matches;
    }

    const int n = static_cast<int>(tracks.size());
    const int m = static_cast<int>(det_indices.size());
    const int size = std::max(n, m);

    constexpr float kLargeCost = 1e6f;
    std::vector<std::vector<float>> cost(size, std::vector<float>(size, 1.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            const float score = iou(tracks[i]->bbox, detections[det_indices[j]].bbox);
            cost[i][j] = (score >= iou_thresh) ? (1.0f - score) : kLargeCost;
        }
    }

    // Hungarian algorithm (min-cost assignment) with 1-indexed work arrays.
    std::vector<float> u(size + 1, 0.0f), v(size + 1, 0.0f);
    std::vector<int> p(size + 1, 0), way(size + 1, 0);
    for (int i = 1; i <= size; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(size + 1, std::numeric_limits<float>::max());
        std::vector<bool> used(size + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            float delta = std::numeric_limits<float>::max();
            int j1 = 0;
            for (int j = 1; j <= size; ++j) {
                if (used[j]) {
                    continue;
                }
                const float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= size; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assigned_track_for_det(size, -1);
    for (int j = 1; j <= size; ++j) {
        if (p[j] > 0) {
            assigned_track_for_det[j - 1] = p[j] - 1;
        }
    }
    for (int det_col = 0; det_col < m; ++det_col) {
        const int track_row = assigned_track_for_det[det_col];
        if (track_row < 0 || track_row >= n) {
            continue;
        }
        if (cost[track_row][det_col] >= kLargeCost) {
            continue;
        }
        matches.emplace_back(track_row, det_col);
    }
    return matches;
}

void ByteTrackTracker::update(uint64_t /*frame_seq*/, std::vector<Detection>& detections) {
    std::vector<int> high_det_idxs;
    std::vector<int> low_det_idxs;
    high_det_idxs.reserve(detections.size());
    low_det_idxs.reserve(detections.size());
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (detections[i].confidence >= cfg_.high_det_thresh) {
            high_det_idxs.push_back(i);
        } else if (detections[i].confidence >= cfg_.low_det_thresh) {
            low_det_idxs.push_back(i);
        }
    }

    std::vector<TrackState*> candidate_tracks;
    candidate_tracks.reserve(tracks_.size());
    for (auto& t : tracks_) {
        if (t.status == TrackStatus::Removed) {
            continue;
        }
        // Constant-velocity prediction.
        t.bbox.x1 += t.vx;
        t.bbox.x2 += t.vx;
        t.bbox.y1 += t.vy;
        t.bbox.y2 += t.vy;
        t.age += 1;
        t.lost += 1;
        if (t.status == TrackStatus::Tracked || t.status == TrackStatus::Lost) {
            candidate_tracks.push_back(&t);
        }
    }

    const auto high_matches = linearAssign(candidate_tracks, high_det_idxs, detections, cfg_.match_iou_thresh);
    std::unordered_set<int> matched_candidate_track_idxs;
    std::unordered_set<int> matched_high_det_list_idxs;
    for (const auto& [ti, di] : high_matches) {
        auto& track = *candidate_tracks[ti];
        auto& det = detections[high_det_idxs[di]];
        const float prev_cx = centerX(track.bbox) - track.vx;
        const float prev_cy = centerY(track.bbox) - track.vy;
        track.vx = centerX(det.bbox) - prev_cx;
        track.vy = centerY(det.bbox) - prev_cy;
        track.bbox = det.bbox;
        track.hits += 1;
        track.lost = 0;
        track.status = TrackStatus::Tracked;
        if (!track.confirmed && track.hits >= cfg_.min_hits_to_confirm) {
            track.confirmed = true;
        }
        if (track.confirmed) {
            det.track_id = track.id;
        }
        matched_candidate_track_idxs.insert(ti);
        matched_high_det_list_idxs.insert(di);
    }

    std::vector<TrackState*> unmatched_after_high;
    unmatched_after_high.reserve(candidate_tracks.size());
    for (int i = 0; i < static_cast<int>(candidate_tracks.size()); ++i) {
        if (matched_candidate_track_idxs.count(i) == 0) {
            unmatched_after_high.push_back(candidate_tracks[i]);
        }
    }
    const auto low_matches = linearAssign(unmatched_after_high, low_det_idxs, detections, cfg_.match_iou_thresh);
    std::unordered_set<TrackState*> matched_tracks;
    for (const auto& [ti, _] : high_matches) {
        matched_tracks.insert(candidate_tracks[ti]);
    }
    for (const auto& [ti, di] : low_matches) {
        auto& track = *unmatched_after_high[ti];
        auto& det = detections[low_det_idxs[di]];
        const float prev_cx = centerX(track.bbox) - track.vx;
        const float prev_cy = centerY(track.bbox) - track.vy;
        track.vx = centerX(det.bbox) - prev_cx;
        track.vy = centerY(det.bbox) - prev_cy;
        track.bbox = det.bbox;
        track.hits += 1;
        track.lost = 0;
        track.status = TrackStatus::Tracked;
        if (!track.confirmed && track.hits >= cfg_.min_hits_to_confirm) {
            track.confirmed = true;
        }
        if (track.confirmed) {
            det.track_id = track.id;
        }
        matched_tracks.insert(&track);
    }

    for (auto& track : tracks_) {
        if (track.status == TrackStatus::Removed) {
            continue;
        }
        if (matched_tracks.count(&track) != 0) {
            continue;
        }
        if (track.lost > cfg_.max_lost_frames) {
            track.status = TrackStatus::Removed;
        } else {
            track.status = TrackStatus::Lost;
        }
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
        t.status = TrackStatus::Tracked;
        t.confirmed = (cfg_.min_hits_to_confirm <= 1);
        if (t.confirmed) {
            detections[high_det_idxs[i]].track_id = t.id;
        }
        tracks_.push_back(t);
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const TrackState& t) { return t.status == TrackStatus::Removed; }),
        tracks_.end());
}

} // namespace infer
