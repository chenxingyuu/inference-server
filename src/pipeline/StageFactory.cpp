#include "pipeline/StageFactory.h"

#include "pipeline/stages/ArchiveRawStage.h"
#include "pipeline/stages/DrawAndStreamStage.h"
#include "pipeline/stages/InferEngineWorkerStage.h"
#include "pipeline/stages/SinkFfplayStage.h"
#include "pipeline/stages/JoinByFrameStage.h"
#include "pipeline/stages/PassthroughStage.h"
#include "pipeline/stages/SinkKafkaStage.h"
#include "pipeline/stages/SourceFileStage.h"
#include "pipeline/stages/SourceRtspStage.h"
#include "pipeline/stages/TrackByteTrackStage.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace infer {

namespace {

// Reject URLs containing control characters (newlines, null bytes, etc.) that
// could be injected into a popen() shell command string.
void validateShellSafeUrl(const std::string& url, const std::string& stage_type) {
    for (unsigned char c : url) {
        if (c < 0x20 || c == 0x7f) {
            throw std::runtime_error(stage_type + " output_url contains illegal control character (0x"
                                     + std::to_string(static_cast<int>(c)) + ")");
        }
    }
}

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

std::string getStringWithDefault(
    const std::map<std::string, std::string>& kv,
    const std::string& key,
    const std::string& default_value) {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.empty()) return default_value;
    return it->second;
}

} // namespace

std::unique_ptr<IStage> StageFactory::create(const StageConfig& cfg, const Context& ctx) {
    if (cfg.type == "source.rtsp") {
        return std::make_unique<SourceRtspStage>(
            cfg.id,
            ctx.source,
            ctx.ingest_sample_fps,
            ctx.ingest_sampling_mode,
            ctx.ingest_use_hwdec,
            ctx.gpu_alloc,
            ctx.ingest_use_ascend_dvpp,
            ctx.ingest_ascend_device_id);
    }
    if (cfg.type == "source.file") {
        bool loop = getStringWithDefault(cfg.with, "loop", "false") == "true";
        return std::make_unique<SourceFileStage>(cfg.id, ctx.source, ctx.ingest_sample_fps, ctx.ingest_sampling_mode, ctx.ingest_use_hwdec, loop);
    }
    if (cfg.type == "decode.ffmpeg" || cfg.type == "preprocess.yolo" || cfg.type == "postprocess.yolo") {
        return std::make_unique<PassthroughStage>(cfg.id);
    }
    if (cfg.type == "archive.raw") {
        return std::make_unique<ArchiveRawStage>(
            cfg.id,
            ctx.frame_archiver,
            ctx.app_config.frame_archive.allow_gpu_frames);
    }
    if (cfg.type == "infer.engine") {
        auto model_id_it = cfg.with.find("model_id");
        if (model_id_it == cfg.with.end()) throw std::runtime_error("infer.engine requires with.model_id");
        const auto* model_cfg = ctx.app_config.findModel(model_id_it->second);
        if (!model_cfg) throw std::runtime_error("infer.engine model not found: " + model_id_it->second);
        return std::make_unique<InferEngineWorkerStage>(
            cfg.id,
            *model_cfg,
            [](const ModelConfig& c) { return createBackend(c); },
            [](const ModelConfig& c) { return createDecoder(c); });
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
    if (cfg.type == "sink.publish") {
        return std::make_unique<SinkKafkaStage>(cfg.id, ctx.publisher);
    }
    if (cfg.type == "sink.ffplay") {
        SinkFfplayConfig ff_cfg;
        ff_cfg.fps = getFloatWithDefault(cfg.with, "fps", static_cast<float>(ff_cfg.fps));
        ff_cfg.queue_capacity = getIntWithDefault(cfg.with, "queue_capacity", ff_cfg.queue_capacity);
        ff_cfg.reconnect_initial_ms = getIntWithDefault(cfg.with, "reconnect_initial_ms", ff_cfg.reconnect_initial_ms);
        ff_cfg.reconnect_max_ms = getIntWithDefault(cfg.with, "reconnect_max_ms", ff_cfg.reconnect_max_ms);
        ff_cfg.draw_conf_thresh = getFloatWithDefault(cfg.with, "draw_conf_thresh", ff_cfg.draw_conf_thresh);
        ff_cfg.line_thickness = getIntWithDefault(cfg.with, "line_thickness", ff_cfg.line_thickness);
        const auto drop_policy = getStringWithDefault(cfg.with, "drop_policy", "drop_oldest");
        if (drop_policy == "drop_oldest") {
            ff_cfg.drop_policy = StreamDropPolicy::DropOldest;
        } else if (drop_policy == "drop_newest") {
            ff_cfg.drop_policy = StreamDropPolicy::DropNewest;
        } else {
            throw std::runtime_error("sink.ffplay drop_policy must be one of: drop_oldest, drop_newest");
        }
        if (ff_cfg.queue_capacity < 1) {
            throw std::runtime_error("sink.ffplay queue_capacity must be >= 1");
        }
        if (ff_cfg.fps <= 0) {
            throw std::runtime_error("sink.ffplay fps must be > 0");
        }
        if (ff_cfg.reconnect_initial_ms < 1 || ff_cfg.reconnect_max_ms < ff_cfg.reconnect_initial_ms) {
            throw std::runtime_error("sink.ffplay reconnect settings invalid");
        }
        return std::make_unique<SinkFfplayStage>(cfg.id, ff_cfg);
    }
    if (cfg.type == "sink.stream") {
        DrawAndStreamConfig stream_cfg;
        stream_cfg.output_url = getStringWithDefault(cfg.with, "output_url", "");
        stream_cfg.protocol = getStringWithDefault(cfg.with, "protocol", "rtsp");
        if (stream_cfg.protocol != "rtsp" && stream_cfg.protocol != "rtmp") {
            throw std::runtime_error("sink.stream protocol must be one of: rtsp, rtmp");
        }
        if (stream_cfg.output_url.empty()) {
            throw std::runtime_error("sink.stream requires with.output_url");
        }
        validateShellSafeUrl(stream_cfg.output_url, "sink.stream");
        stream_cfg.fps = getFloatWithDefault(cfg.with, "fps", static_cast<float>(stream_cfg.fps));
        stream_cfg.bitrate_kbps = getIntWithDefault(cfg.with, "bitrate_kbps", stream_cfg.bitrate_kbps);
        stream_cfg.queue_capacity = getIntWithDefault(cfg.with, "queue_capacity", stream_cfg.queue_capacity);
        stream_cfg.reconnect_initial_ms = getIntWithDefault(cfg.with, "reconnect_initial_ms", stream_cfg.reconnect_initial_ms);
        stream_cfg.reconnect_max_ms = getIntWithDefault(cfg.with, "reconnect_max_ms", stream_cfg.reconnect_max_ms);
        stream_cfg.draw_conf_thresh = getFloatWithDefault(cfg.with, "draw_conf_thresh", stream_cfg.draw_conf_thresh);
        stream_cfg.line_thickness = getIntWithDefault(cfg.with, "line_thickness", stream_cfg.line_thickness);
        const auto drop_policy = getStringWithDefault(cfg.with, "drop_policy", "drop_oldest");
        if (drop_policy == "drop_oldest") {
            stream_cfg.drop_policy = StreamDropPolicy::DropOldest;
        } else if (drop_policy == "drop_newest") {
            stream_cfg.drop_policy = StreamDropPolicy::DropNewest;
        } else {
            throw std::runtime_error("sink.stream drop_policy must be one of: drop_oldest, drop_newest");
        }
        if (stream_cfg.queue_capacity < 1) {
            throw std::runtime_error("sink.stream queue_capacity must be >= 1");
        }
        if (stream_cfg.fps <= 0) {
            throw std::runtime_error("sink.stream fps must be > 0");
        }
        if (stream_cfg.reconnect_initial_ms < 1 || stream_cfg.reconnect_max_ms < stream_cfg.reconnect_initial_ms) {
            throw std::runtime_error("sink.stream reconnect settings invalid");
        }
        return std::make_unique<DrawAndStreamStage>(cfg.id, stream_cfg);
    }
    throw std::runtime_error("unsupported stage type: " + cfg.type);
}

} // namespace infer
