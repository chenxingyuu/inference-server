#include "pipeline/StageFactory.h"

#include "pipeline/stages/ArchiveRawStage.h"
#include "pipeline/stages/InferEngineStage.h"
#include "pipeline/stages/JoinByFrameStage.h"
#include "pipeline/stages/PassthroughStage.h"
#include "pipeline/stages/SinkKafkaStage.h"
#include "pipeline/stages/SourceRtspStage.h"
#include "pipeline/stages/TrackByteTrackStage.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace infer {

namespace {

int getIntWithDefault(const std::map<std::string, std::string>& kv, const std::string& key, int default_value) {
    auto it = kv.find(key);
    if (it == kv.end()) return default_value;
    try {
        return std::stoi(it->second);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer value for '" + key + "': " + it->second);
    }
}

float getFloatWithDefault(const std::map<std::string, std::string>& kv, const std::string& key, float default_value) {
    auto it = kv.find(key);
    if (it == kv.end()) return default_value;
    try {
        return std::stof(it->second);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float value for '" + key + "': " + it->second);
    }
}

} // namespace

std::unique_ptr<IStage> StageFactory::create(const StageConfig& cfg, const Context& ctx) {
    if (cfg.type == "source.rtsp") {
        return std::make_unique<SourceRtspStage>(cfg.id, ctx.source);
    }
    if (cfg.type == "decode.ffmpeg" || cfg.type == "preprocess.yolo" || cfg.type == "postprocess.yolo") {
        return std::make_unique<PassthroughStage>(cfg.id);
    }
    if (cfg.type == "archive.raw") {
        return std::make_unique<ArchiveRawStage>(cfg.id, ctx.frame_archiver);
    }
    if (cfg.type == "infer.engine") {
        auto model_id_it = cfg.with.find("model_id");
        if (model_id_it == cfg.with.end()) throw std::runtime_error("infer.engine requires with.model_id");
        const auto* model_cfg = ctx.app_config.findModel(model_id_it->second);
        if (!model_cfg) throw std::runtime_error("infer.engine model not found: " + model_id_it->second);
        return std::make_unique<InferEngineStage>(
            cfg.id, *model_cfg, createBackend(*model_cfg), createDecoder(*model_cfg));
    }
    if (cfg.type == "track.bytetrack") {
        ByteTrackConfig bt;
        bt.high_det_thresh = getFloatWithDefault(cfg.with, "high_det_thresh", bt.high_det_thresh);
        bt.low_det_thresh = getFloatWithDefault(cfg.with, "low_det_thresh", bt.low_det_thresh);
        bt.match_iou_thresh = getFloatWithDefault(cfg.with, "match_iou_thresh", bt.match_iou_thresh);
        bt.min_hits_to_confirm = getIntWithDefault(cfg.with, "min_hits_to_confirm", bt.min_hits_to_confirm);
        bt.max_lost_frames = getIntWithDefault(cfg.with, "max_lost_frames", bt.max_lost_frames);
        return std::make_unique<TrackByteTrackStage>(cfg.id, bt);
    }
    if (cfg.type == "join.byFrameId") {
        return std::make_unique<JoinByFrameStage>(cfg.id);
    }
    if (cfg.type == "sink.kafka") {
        return std::make_unique<SinkKafkaStage>(cfg.id, ctx.publisher);
    }
    throw std::runtime_error("unsupported stage type: " + cfg.type);
}

} // namespace infer
