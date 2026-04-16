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
    std::shared_ptr<Frame> frame;
    std::optional<InferResult> infer_result;
    std::optional<ArchiveInfo> archive_info;
};

} // namespace infer
