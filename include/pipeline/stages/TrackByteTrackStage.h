#pragma once

#include "common/Config.h"
#include "pipeline/IStage.h"
#include "tracker/TrackerManager.h"
#include <string>

namespace infer {

class TrackByteTrackStage final : public IStage {
public:
    TrackByteTrackStage(std::string id, ByteTrackConfig cfg);

    std::string id() const override;
    void process(const EventEnvelope& input, const EmitFn& emit) override;

private:
    std::string id_;
    ByteTrackConfig bt_cfg_;
    TrackerManager tracker_manager_;
};

} // namespace infer
