#include "pipeline/stages/ArchiveRawStage.h"

namespace infer {

ArchiveRawStage::ArchiveRawStage(std::string id, std::shared_ptr<FrameArchiver> archiver)
    : id_(std::move(id)), archiver_(std::move(archiver)) {}

std::string ArchiveRawStage::id() const { return id_; }

void ArchiveRawStage::process(const EventEnvelope& input, const EmitFn& emit) {
    EventEnvelope out = input;
    if (archiver_ && input.frame && !input.frame->is_gpu) {
        auto result = archiver_->submit(input.frame->meta, &input.frame->image);
        out.archive_info = ArchiveInfo{result.local_path, result.frame_url, result.upload_state};
    } else {
        // GPU frames or no archiver: set disabled so JoinByFrameStage is not blocked.
        out.archive_info = ArchiveInfo{"", "", "disabled"};
    }
    emit(out);
}

} // namespace infer
