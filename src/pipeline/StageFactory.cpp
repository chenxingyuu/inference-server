#include "pipeline/StageFactory.h"

#include "common/Logger.h"
#include "stream/FFmpegDecoder.h"
#include "tracker/TrackerManager.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace infer {

namespace {

double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

uint64_t nowSteadyNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

double nsToMs(uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

class SourceRtspStage final : public IStage {
public:
    SourceRtspStage(std::string id, const PipelineSourceConfig& src)
        : id_(std::move(id)), source_(src), max_queue_size_(std::max(32, src.sample_fps * 2)) {}

    std::string id() const override { return id_; }
    bool isSource() const override { return true; }

    void start() override {
        if (running_) return;
        running_ = true;
        StreamConfig cfg;
        cfg.id = source_.id;
        cfg.url = source_.url;
        cfg.sample_fps = source_.sample_fps;
        cfg.reconnect_delay_ms = source_.reconnect_delay_ms;
        cfg.max_reconnect_delay_ms = source_.max_reconnect_delay_ms;
        cfg.degraded_threshold = source_.degraded_threshold;
        cfg.max_reconnect_attempts = source_.max_reconnect_attempts;
        cfg.use_hwdec = source_.use_hwdec;
        decoder_.start(cfg, [this](Frame frame) {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() >= max_queue_size_) {
                queue_.pop_front();
            }
            queue_.push_back(std::move(frame));
        });
    }

    void stop() override {
        running_ = false;
        decoder_.stop();
    }

    void process(const EventEnvelope&, const EmitFn& emit) override {
        Frame frame;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                return;
            }
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        EventEnvelope out;
        out.event_id = frame.meta.stream_id + ":" + std::to_string(frame.meta.frame_seq);
        out.stream_id = frame.meta.stream_id;
        out.frame_seq = frame.meta.frame_seq;
        out.frame = std::make_shared<Frame>(std::move(frame));
        emit(out);
    }

private:
    std::string id_;
    PipelineSourceConfig source_;
    FFmpegDecoder decoder_;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::deque<Frame> queue_;
    std::size_t max_queue_size_{64};
};

class PassthroughStage final : public IStage {
public:
    explicit PassthroughStage(std::string id) : id_(std::move(id)) {}
    std::string id() const override { return id_; }
    void process(const EventEnvelope& input, const EmitFn& emit) override { emit(input); }
private:
    std::string id_;
};

class ArchiveRawStage final : public IStage {
public:
    ArchiveRawStage(std::string id, std::shared_ptr<FrameArchiver> archiver)
        : id_(std::move(id)), archiver_(std::move(archiver)) {}
    std::string id() const override { return id_; }
    void process(const EventEnvelope& input, const EmitFn& emit) override {
        EventEnvelope out = input;
        if (archiver_ && input.frame && !input.frame->is_gpu) {
            auto result = archiver_->submit(input.frame->meta, &input.frame->image);
            out.archive_info = ArchiveInfo{result.local_path, result.frame_url, result.upload_state};
        }
        emit(out);
    }
private:
    std::string id_;
    std::shared_ptr<FrameArchiver> archiver_;
};

class InferEngineStage final : public IStage {
public:
    InferEngineStage(std::string id,
                     ModelConfig model_cfg,
                     std::unique_ptr<IInferBackend> backend,
                     std::unique_ptr<IYOLODecoder> decoder)
        : id_(std::move(id))
        , model_cfg_(std::move(model_cfg))
        , backend_(std::move(backend))
        , decoder_(std::move(decoder)) {
        batch_deadline_ = std::chrono::steady_clock::now() + std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    }

    std::string id() const override { return id_; }

    void start() override {
        // Load lazily on first inference.
    }
    void stop() override {
        if (loaded_) {
            backend_->unloadModel();
            loaded_ = false;
        }
    }

    void process(const EventEnvelope& input, const EmitFn& emit) override {
        if (input.frame) {
            pending_events_.push_back(input);
        }
        const auto now = std::chrono::steady_clock::now();
        if (pending_events_.empty()) return;
        const int effective_batch_size = fallback_to_single_infer_
            ? 1
            : std::max(1, std::min(model_cfg_.batch_size, backend_->maxBatchSize()));
        const bool hit_batch = static_cast<int>(pending_events_.size()) >= effective_batch_size;
        const bool hit_deadline = now >= batch_deadline_;
        if (!hit_batch && !hit_deadline) return;

        const int flush_count = std::min(static_cast<int>(pending_events_.size()), effective_batch_size);
        try {
            // 延迟加载模型：仅在首次需要推理时执行一次。
            if (!loaded_) {
                backend_->loadModel(model_cfg_);
                loaded_ = true;
            }
            Batch batch;
            batch.is_gpu = pending_events_.front().frame->is_gpu;
            for (int i = 0; i < flush_count; ++i) {
                const auto& ev = pending_events_[i];
                batch.metas.push_back(ev.frame->meta);
                if (batch.is_gpu) batch.gpu_frames.push_back(ev.frame->gpu_buf);
                else batch.frames.push_back(ev.frame->image);
            }
            std::vector<float> output;
            const auto infer_start = std::chrono::steady_clock::now();
            backend_->infer(batch, output);
            const auto infer_end_tp = std::chrono::steady_clock::now();
            const double infer_ms = std::chrono::duration<double, std::milli>(infer_end_tp - infer_start).count();

            const auto decode_start = std::chrono::steady_clock::now();
            auto decoded = decoder_->decode(output.data(), batch.size(), model_cfg_.input_shape,
                                            model_cfg_.conf_thresh, model_cfg_.nms_thresh);
            const double decode_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();

            const uint64_t infer_finish_mono_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            const double infer_ts = nowEpoch();
            for (int i = 0; i < batch.size(); ++i) {
                EventEnvelope out = pending_events_[i];
                InferResult result;
                result.stream_id = out.stream_id;
                result.frame_seq = out.frame_seq;
                result.frame_ts = out.frame->meta.capture_ts;
                result.frame_mono_ns = out.frame->meta.capture_mono_ns;
                result.infer_ts = infer_ts;
                result.infer_ms = infer_ms;
                result.decode_ms = decode_ms;
                if (result.frame_mono_ns != 0) {
                    const uint64_t batch_start_mono_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            infer_start.time_since_epoch()).count());
                    result.queue_latency_ms = nsToMs(batch_start_mono_ns - result.frame_mono_ns);
                    result.latency_ms = nsToMs(infer_finish_mono_ns - result.frame_mono_ns);
                } else {
                    // Fallback: epoch-based duration (may be affected by system clock jumps).
                    result.latency_ms = (infer_ts - result.frame_ts) * 1000.0;
                    result.queue_latency_ms = 0.0;
                }
                result.model_id = model_cfg_.id;
                if (i < static_cast<int>(decoded.size())) {
                    result.detections = std::move(decoded[i]);
                }
                out.infer_result = std::move(result);
                emit(out);
            }
        } catch (const std::exception& e) {
            const std::string err = e.what();
            if (err.find("invalid dimensions") != std::string::npos && flush_count > 1) {
                LOG_WARN("InferEngineStage[{}]: batch inference failed with dimension mismatch, fallback to bs=1",
                         id_);
                fallback_to_single_infer_ = true;
                for (int i = 0; i < flush_count; ++i) {
                    try {
                        Batch one;
                        one.is_gpu = pending_events_[i].frame->is_gpu;
                        one.metas.push_back(pending_events_[i].frame->meta);
                        if (one.is_gpu) one.gpu_frames.push_back(pending_events_[i].frame->gpu_buf);
                        else one.frames.push_back(pending_events_[i].frame->image);

                        std::vector<float> output;
                        const auto infer_start = std::chrono::steady_clock::now();
                        backend_->infer(one, output);
                        const auto infer_end_tp = std::chrono::steady_clock::now();
                        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end_tp - infer_start).count();

                        const auto decode_start = std::chrono::steady_clock::now();
                        auto decoded = decoder_->decode(output.data(), 1, model_cfg_.input_shape,
                                                        model_cfg_.conf_thresh, model_cfg_.nms_thresh);
                        const double decode_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - decode_start).count();

                        EventEnvelope out = pending_events_[i];
                        InferResult result;
                        result.stream_id = out.stream_id;
                        result.frame_seq = out.frame_seq;
                        result.frame_ts = out.frame->meta.capture_ts;
                        result.frame_mono_ns = out.frame->meta.capture_mono_ns;
                        result.infer_ts = nowEpoch();
                        result.infer_ms = infer_ms;
                        result.decode_ms = decode_ms;
                        if (result.frame_mono_ns != 0) {
                            const uint64_t batch_start_mono_ns = static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    infer_start.time_since_epoch()).count());
                            const uint64_t infer_finish_mono_ns = nowSteadyNs();
                            result.queue_latency_ms = nsToMs(batch_start_mono_ns - result.frame_mono_ns);
                            result.latency_ms = nsToMs(infer_finish_mono_ns - result.frame_mono_ns);
                        } else {
                            result.latency_ms = (result.infer_ts - result.frame_ts) * 1000.0;
                            result.queue_latency_ms = 0.0;
                        }
                        result.model_id = model_cfg_.id;
                        if (!decoded.empty()) result.detections = std::move(decoded[0]);
                        out.infer_result = std::move(result);
                        emit(out);
                    } catch (const std::exception& one_e) {
                        LOG_ERROR("InferEngineStage[{}]: fallback bs=1 failed: {}", id_, one_e.what());
                    }
                }
            } else {
                LOG_ERROR("InferEngineStage[{}]: inference failed: {}", id_, err);
            }
        }
        pending_events_.erase(pending_events_.begin(), pending_events_.begin() + flush_count);
        batch_deadline_ = std::chrono::steady_clock::now() + std::chrono::microseconds(model_cfg_.max_queue_delay_us);
    }

private:
    std::string id_;
    ModelConfig model_cfg_;
    std::unique_ptr<IInferBackend> backend_;
    std::unique_ptr<IYOLODecoder> decoder_;
    std::vector<EventEnvelope> pending_events_;
    std::chrono::steady_clock::time_point batch_deadline_;
    bool fallback_to_single_infer_{false};
    bool loaded_{false};
};

class TrackByteTrackStage final : public IStage {
public:
    TrackByteTrackStage(std::string id, ByteTrackConfig cfg)
        : id_(std::move(id)), bt_cfg_(cfg) {}
    std::string id() const override { return id_; }
    void process(const EventEnvelope& input, const EmitFn& emit) override {
        EventEnvelope out = input;
        if (out.infer_result) {
            tracker_manager_.apply(out.infer_result->stream_id, TrackerType::ByteTrack, bt_cfg_,
                                   out.infer_result->frame_seq, out.infer_result->detections);
        }
        emit(out);
    }
private:
    std::string id_;
    ByteTrackConfig bt_cfg_;
    TrackerManager tracker_manager_;
};

class JoinByFrameStage final : public IStage {
public:
    explicit JoinByFrameStage(std::string id) : id_(std::move(id)) {}
    std::string id() const override { return id_; }

    void process(const EventEnvelope& input, const EmitFn& emit) override {
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

private:
    struct JoinState {
        std::optional<InferResult> infer;
        std::optional<ArchiveInfo> archive;
        std::chrono::steady_clock::time_point created_at{};
    };
    std::string id_;
    std::mutex mu_;
    std::unordered_map<std::string, JoinState> pending_;
};

class SinkKafkaStage final : public IStage {
public:
    SinkKafkaStage(std::string id, IPublisher& publisher)
        : id_(std::move(id)), publisher_(publisher) {}
    std::string id() const override { return id_; }
    void process(const EventEnvelope& input, const EmitFn&) override {
        if (input.infer_result) publisher_.publish(*input.infer_result);
    }
private:
    std::string id_;
    IPublisher& publisher_;
};

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
