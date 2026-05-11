#include "pipeline/stages/SahiMergeStage.h"

#include "common/Logger.h"
#include "pipeline/stages/SahiRoiRegistry.h"

#include <algorithm>

namespace infer {

SahiMergeStage::SahiMergeStage(std::string id, SahiMergeConfig cfg)
    : id_(std::move(id)), cfg_(cfg) {}

std::string SahiMergeStage::id() const { return id_; }

float SahiMergeStage::iou(const BBox& a, const BBox& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float uni = a.area() + b.area() - inter;
    return uni <= 0.0f ? 0.0f : (inter / uni);
}

std::vector<Detection> SahiMergeStage::runNms(const std::vector<Detection>& detections) const {
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.confidence > rhs.confidence;
    });

    std::vector<Detection> kept;
    std::vector<bool> removed(sorted.size(), false);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(sorted[i]);
        for (std::size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) continue;
            if (sorted[i].class_id != sorted[j].class_id) continue;
            if (iou(sorted[i].bbox, sorted[j].bbox) >= cfg_.merge_iou) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

void SahiMergeStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (!input.sahi_tile.has_value()) {
        if (input.infer_result.has_value()) {
            SahiRoiRegistry::update(input.stream_id, input.infer_result->frame_seq, input.infer_result->detections);
        }
        emit(input);
        return;
    }
    if (!input.infer_result.has_value()) return;

    const auto& tile = *input.sahi_tile;
    const PendingKey key{input.stream_id, tile.parent_frame_seq};
    std::optional<EventEnvelope> completed;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = pending_.begin(); it != pending_.end();) {
            const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.created_at).count();
            if (age_ms > cfg_.stale_timeout_ms) {
                LOG_WARN("SahiMergeStage[{}]: drop stale pending stream={} parent_frame_seq={} received={}/{}",
                         id_,
                         it->second.stream_id,
                         it->second.parent_frame_seq,
                         it->second.received_tiles,
                         it->second.expected_tiles);
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }

        auto& state = pending_[key];
        if (state.received_tiles == 0) {
            state.created_at = now;
            state.parent_frame = tile.parent_frame;
            state.template_result = input.infer_result;
            state.expected_tiles = std::max(1, tile.expected_tile_count);
            state.parent_frame_seq = tile.parent_frame_seq;
            state.stream_id = input.stream_id;
            state.event_id = input.event_id;
            LOG_DEBUG("SahiMergeStage[{}]: stream={} parent_frame_seq={} begin aggregation expected_tiles={} mode={}",
                      id_,
                      input.stream_id,
                      tile.parent_frame_seq,
                      state.expected_tiles,
                      tile.is_full_pass ? "full" : "roi");
        }
        const std::size_t tile_det_count = input.infer_result->detections.size();
        for (const auto& det : input.infer_result->detections) {
            Detection mapped = det;
            mapped.bbox.x1 += static_cast<float>(tile.tile_x);
            mapped.bbox.y1 += static_cast<float>(tile.tile_y);
            mapped.bbox.x2 += static_cast<float>(tile.tile_x);
            mapped.bbox.y2 += static_cast<float>(tile.tile_y);
            state.detections.push_back(std::move(mapped));
        }
        state.received_tiles++;
        LOG_DEBUG("SahiMergeStage[{}]: stream={} parent_frame_seq={} recv_tile={}/{} tile_frame_seq={} tile_rect=({},{} {}x{}) tile_dets={} aggregated_dets={}",
                  id_,
                  input.stream_id,
                  tile.parent_frame_seq,
                  state.received_tiles,
                  state.expected_tiles,
                  tile.tile_frame_seq,
                  tile.tile_x,
                  tile.tile_y,
                  tile.tile_width,
                  tile.tile_height,
                  tile_det_count,
                  state.detections.size());

        if (state.received_tiles >= state.expected_tiles && state.template_result.has_value()) {
            EventEnvelope out = input;
            out.event_id = state.event_id;
            out.sahi_tile.reset();
            out.frame = state.parent_frame ? state.parent_frame : input.frame;
            InferResult merged = *state.template_result;
            merged.frame_seq = state.parent_frame_seq;
            merged.detections = runNms(state.detections);
            if (out.frame) {
                merged.frame_mono_ns = out.frame->meta.capture_mono_ns;
            }
            out.infer_result = std::move(merged);
            LOG_DEBUG("SahiMergeStage[{}]: stream={} parent_frame_seq={} merge_complete raw_dets={} merged_dets={}",
                      id_,
                      state.stream_id,
                      state.parent_frame_seq,
                      state.detections.size(),
                      out.infer_result->detections.size());
            pending_.erase(key);
            completed = std::move(out);
        }
    }

    if (completed && completed->infer_result.has_value()) {
        SahiRoiRegistry::update(completed->stream_id, completed->infer_result->frame_seq, completed->infer_result->detections);
        emit(*completed);
    }
}

} // namespace infer
