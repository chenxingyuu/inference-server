#include "pipeline/stages/JoinByFrameStage.h"

namespace infer {

JoinByFrameStage::JoinByFrameStage(std::string id) : id_(std::move(id)) {}

std::string JoinByFrameStage::id() const { return id_; }

void JoinByFrameStage::process(const EventEnvelope& input, const EmitFn& emit) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (now - it->second.created_at > std::chrono::seconds(5)) it = pending_.erase(it);
        else ++it;
    }
    auto& state = pending_[input.event_id];
    if (state.created_at.time_since_epoch().count() == 0) {
        state.created_at = now;
    }
    if (input.infer_result) state.infer = input.infer_result;
    if (input.archive_info) state.archive = input.archive_info;
    if (state.infer && state.archive) {
        EventEnvelope out = input;
        out.infer_result = state.infer;
        out.infer_result->frame_local_path = state.archive->local_path;
        out.infer_result->frame_url = state.archive->frame_url;
        out.infer_result->frame_upload_state = state.archive->upload_state;
        pending_.erase(input.event_id);
        emit(out);
    }
}

} // namespace infer
