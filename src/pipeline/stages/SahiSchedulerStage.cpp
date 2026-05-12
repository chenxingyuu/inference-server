#include "pipeline/stages/SahiSchedulerStage.h"

#include "common/Logger.h"
#include "pipeline/stages/SahiRoiRegistry.h"

#include <algorithm>
#include <cmath>

namespace infer {

SahiSchedulerStage::SahiSchedulerStage(std::string id, SahiSchedulerConfig cfg)
    : id_(std::move(id)), cfg_(cfg) {}

std::string SahiSchedulerStage::id() const { return id_; }

std::vector<int> SahiSchedulerStage::makeAxisStarts(int full, int tile, float overlap) const {
    const int safe_tile = std::max(1, std::min(tile, full));
    const int overlap_px = static_cast<int>(std::round(static_cast<double>(safe_tile) * overlap));
    const int stride = std::max(1, safe_tile - overlap_px);
    std::vector<int> starts;
    for (int x = 0; x < full; x += stride) {
        const int start = std::min(x, full - safe_tile);
        if (starts.empty() || starts.back() != start) starts.push_back(start);
        if (start + safe_tile >= full) break;
    }
    return starts;
}

std::vector<cv::Rect> SahiSchedulerStage::makeFullTiles(int full_w, int full_h) const {
    const auto xs = makeAxisStarts(full_w, cfg_.tile_width, cfg_.x_overlap_ratio);
    const auto ys = makeAxisStarts(full_h, cfg_.tile_height, cfg_.y_overlap_ratio);
    std::vector<cv::Rect> out;
    out.reserve(xs.size() * ys.size());
    for (int y : ys) {
        const int h = std::min(cfg_.tile_height, full_h - y);
        for (int x : xs) {
            const int w = std::min(cfg_.tile_width, full_w - x);
            out.emplace_back(x, y, w, h);
        }
    }
    return out;
}

std::vector<cv::Rect> SahiSchedulerStage::makeRoiTiles(int full_w, int full_h, const std::string& stream_id) const {
    const auto roi_snapshot = SahiRoiRegistry::get(stream_id);
    if (!roi_snapshot || roi_snapshot->detections.empty()) return {};

    std::vector<cv::Rect> out;
    for (const auto& det : roi_snapshot->detections) {
        const float det_w = std::max(1.0f, det.bbox.x2 - det.bbox.x1);
        const float det_h = std::max(1.0f, det.bbox.y2 - det.bbox.y1);
        const float cx = (det.bbox.x1 + det.bbox.x2) * 0.5f;
        const float cy = (det.bbox.y1 + det.bbox.y2) * 0.5f;
        const float want_w = std::max(static_cast<float>(cfg_.min_roi_width), det_w * cfg_.roi_expand_ratio);
        const float want_h = std::max(static_cast<float>(cfg_.min_roi_height), det_h * cfg_.roi_expand_ratio);
        int x = static_cast<int>(std::floor(cx - want_w * 0.5f));
        int y = static_cast<int>(std::floor(cy - want_h * 0.5f));
        int w = static_cast<int>(std::ceil(want_w));
        int h = static_cast<int>(std::ceil(want_h));
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, full_w - x);
        h = std::min(h, full_h - y);
        if (w > 0 && h > 0) out.emplace_back(x, y, w, h);
    }
    if (out.empty()) return out;

    // Keep deterministic ordering and dedup near-identical windows.
    std::sort(out.begin(), out.end(), [](const cv::Rect& a, const cv::Rect& b) {
        if (a.y != b.y) return a.y < b.y;
        if (a.x != b.x) return a.x < b.x;
        if (a.width != b.width) return a.width < b.width;
        return a.height < b.height;
    });
    out.erase(std::unique(out.begin(), out.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    }), out.end());
    return out;
}

uint64_t SahiSchedulerStage::nextTileSeqLocked(const std::string& stream_id) {
    auto& s = stream_state_[stream_id];
    return s.next_tile_seq++;
}

void SahiSchedulerStage::sweepStaleStreamsLocked() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_sweep_ < std::chrono::seconds(10)) return;
    last_sweep_ = now;
    for (auto it = stream_state_.begin(); it != stream_state_.end();) {
        if (now - it->second.last_seen > std::chrono::seconds(60)) {
            SahiRoiRegistry::remove(it->first);
            it = stream_state_.erase(it);
        } else {
            ++it;
        }
    }
}

void SahiSchedulerStage::process(const EventEnvelope& input, const EmitFn& emit) {
    if (!input.frame || input.sahi_tile.has_value()) {
        emit(input);
        return;
    }
    if (input.frame->image.empty()) {
        emit(input);
        return;
    }

    const int full_w = input.frame->image.cols;
    const int full_h = input.frame->image.rows;
    if (full_w <= 0 || full_h <= 0) {
        emit(input);
        return;
    }

    bool use_full_pass = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        sweepStaleStreamsLocked();
        auto& s = stream_state_[input.stream_id];
        s.frame_count++;
        s.last_seen = std::chrono::steady_clock::now();
        const int interval = std::max(1, cfg_.full_interval);
        use_full_pass = (interval <= 1) || ((s.frame_count % static_cast<uint64_t>(interval)) == 1);
    }

    bool fallback_single_tile = false;
    if (!use_full_pass) {
        const auto roi_snapshot = SahiRoiRegistry::get(input.stream_id);
        const bool roi_missing_or_stale =
            !roi_snapshot.has_value() ||
            (input.frame->meta.frame_seq > roi_snapshot->frame_seq &&
             (input.frame->meta.frame_seq - roi_snapshot->frame_seq) > static_cast<uint64_t>(std::max(1, cfg_.roi_max_age_frames)));

        if (roi_missing_or_stale) {
            std::lock_guard<std::mutex> lock(mu_);
            auto& s = stream_state_[input.stream_id];
            const uint64_t gap = std::max(1, cfg_.fallback_full_min_gap_frames);
            if (s.frame_count - s.last_fallback_full_frame >= gap) {
                use_full_pass = true;
                s.last_fallback_full_frame = s.frame_count;
                LOG_DEBUG("SahiSchedulerStage[{}]: stream={} frame_seq={} ROI missing/stale, fallback full pass",
                          id_, input.stream_id, input.frame->meta.frame_seq);
            } else {
                fallback_single_tile = true;
                LOG_DEBUG("SahiSchedulerStage[{}]: stream={} frame_seq={} ROI missing/stale, throttled single tile",
                          id_, input.stream_id, input.frame->meta.frame_seq);
            }
        }
    }

    std::vector<cv::Rect> tiles;
    if (fallback_single_tile) {
        tiles.emplace_back(0, 0, full_w, full_h);
    } else {
        tiles = use_full_pass ? makeFullTiles(full_w, full_h) : makeRoiTiles(full_w, full_h, input.stream_id);
        if (tiles.empty()) {
            tiles.emplace_back(0, 0, full_w, full_h);
            LOG_DEBUG("SahiSchedulerStage[{}]: stream={} frame_seq={} empty ROI tiles, single tile fallback",
                      id_, input.stream_id, input.frame->meta.frame_seq);
        }
    }
    if (static_cast<int>(tiles.size()) > cfg_.max_tiles_per_frame) {
        tiles.resize(static_cast<std::size_t>(cfg_.max_tiles_per_frame));
        LOG_WARN("SahiSchedulerStage[{}]: stream={} capped tiles to {}", id_, input.stream_id, cfg_.max_tiles_per_frame);
    }

    const int expected_tiles = static_cast<int>(tiles.size());
    LOG_DEBUG("SahiSchedulerStage[{}]: stream={} frame_seq={} mode={} full_size={}x{} tiles={}",
              id_,
              input.stream_id,
              input.frame->meta.frame_seq,
              use_full_pass ? "full" : "roi",
              full_w,
              full_h,
              expected_tiles);
    for (int i = 0; i < expected_tiles; ++i) {
        const auto& rect = tiles[static_cast<std::size_t>(i)];
        EventEnvelope out = input;
        auto tile_frame = std::make_shared<Frame>();
        tile_frame->image = input.frame->image(rect);
        tile_frame->meta = input.frame->meta;
        tile_frame->meta.orig_width = rect.width;
        tile_frame->meta.orig_height = rect.height;
        {
            std::lock_guard<std::mutex> lock(mu_);
            tile_frame->meta.frame_seq = nextTileSeqLocked(input.stream_id);
        }
        out.frame = tile_frame;
        out.frame_seq = tile_frame->meta.frame_seq;
        out.event_id = input.event_id + "#tile" + std::to_string(i);
        out.infer_result.reset();
        out.sahi_tile = SahiTileInfo{
            input.frame->meta.frame_seq,
            tile_frame->meta.frame_seq,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            full_w,
            full_h,
            use_full_pass,
            expected_tiles,
            (i == 0) ? input.frame : nullptr};
        LOG_DEBUG("SahiSchedulerStage[{}]: stream={} parent_frame_seq={} emit_tile={}/{} rect=({},{} {}x{}) tile_frame_seq={}",
                  id_,
                  input.stream_id,
                  input.frame->meta.frame_seq,
                  i + 1,
                  expected_tiles,
                  rect.x,
                  rect.y,
                  rect.width,
                  rect.height,
                  tile_frame->meta.frame_seq);
        emit(out);
    }
}

} // namespace infer
