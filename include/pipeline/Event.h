#pragma once

#include "common/Types.h"
#include <optional>
#include <string>
#include <memory>

namespace infer {

struct ArchiveInfo {
    std::string local_path;
    std::string frame_url;
    std::string upload_state;
};

struct SahiTileInfo {
    uint64_t parent_frame_seq{0};
    uint64_t tile_frame_seq{0};
    int tile_x{0};
    int tile_y{0};
    int tile_width{0};
    int tile_height{0};
    int full_width{0};
    int full_height{0};
    bool is_full_pass{true};
    int expected_tile_count{1};
    std::shared_ptr<Frame> parent_frame;
};

struct EventEnvelope {
    std::string event_id;
    std::string stream_id;
    uint64_t    frame_seq{0};
    std::optional<std::size_t> source_queue_size;
    std::optional<std::string> ingress_edge;
    std::optional<std::size_t> ingress_edge_queue_size;
    uint64_t    received_at_infer_ns{0};  // set on entry to InferEngineStage::process()
    std::shared_ptr<Frame> frame;
    std::optional<SahiTileInfo> sahi_tile;
    std::optional<InferResult> infer_result;
    std::optional<ArchiveInfo> archive_info;
};

} // namespace infer
