#include "pipeline/stages/JoinByFrameStage.h"

namespace infer {

JoinByFrameStage::JoinByFrameStage(std::string id) : id_(std::move(id)) {}

std::string JoinByFrameStage::id() const { return id_; }

void JoinByFrameStage::process(const EventEnvelope& input, const EmitFn& emit) {
    std::optional<EventEnvelope> completed;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = pending_.begin(); it != pending_.end();) {
            if (now - it->second.created_at > std::chrono::seconds(5)) it = pending_.erase(it);
            else ++it;
        }
        auto& state = pending_[input.event_id];
        if (!state.initialized) {
            state.created_at = now;
            state.initialized = true;
        }
        if (input.infer_result) state.infer = input.infer_result;
        if (input.archive_info) state.archive = input.archive_info;
        if (state.infer && state.archive) {
            EventEnvelope out = input;
            out.infer_result = state.infer;
            out.infer_result->frame_local_path = state.archive->local_path;
            out.infer_result->frame_upload_state = state.archive->upload_state;
            pending_.erase(input.event_id);
            completed = std::move(out);
        }
    }
    // Emit outside the lock to avoid lock-order inversion with EdgeQueue mutex.
    if (completed) emit(*completed);
}

} // namespace infer
