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

struct EventEnvelope {
    std::string event_id;
    std::string stream_id;
    uint64_t    frame_seq{0};
    std::optional<std::size_t> source_queue_size;
    std::optional<std::string> ingress_edge;
    std::optional<std::size_t> ingress_edge_queue_size;
    uint64_t    received_at_infer_ns{0};  // set on entry to InferEngineStage::process()
    std::shared_ptr<Frame> frame;
    std::optional<InferResult> infer_result;
    std::optional<ArchiveInfo> archive_info;
};

} // namespace infer
