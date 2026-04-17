#include "pipeline/stages/TrackByteTrackStage.h"

namespace infer {

TrackByteTrackStage::TrackByteTrackStage(std::string id, ByteTrackConfig cfg)
    : id_(std::move(id)), bt_cfg_(cfg) {}

std::string TrackByteTrackStage::id() const { return id_; }

void TrackByteTrackStage::process(const EventEnvelope& input, const EmitFn& emit) {
    EventEnvelope out = input;
    if (out.infer_result) {
        tracker_manager_.apply(out.infer_result->stream_id, TrackerType::ByteTrack, bt_cfg_,
                               out.infer_result->frame_seq, out.infer_result->detections);
    }
    emit(out);
}

} // namespace infer
