#include "stream/FFmpegDecoder.h"
#include "stream/StopAwareSleep.h"
#include "stream/StreamHealthRegistry.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "publisher/ControlEventBus.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <string>
#include <thread>
#include <mutex>

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

// Route FFmpeg log messages through spdlog so they never appear raw on stderr.
// Level mapping rationale:
//   AV_LOG_ERROR covers many non-fatal codec quirks (e.g. SEI truncation).
//   Real failures are surfaced via return codes, not log messages, so we only
//   need AV_LOG_FATAL+ to appear as errors in our own logger.
void ffmpegLogCallback(void* /*avcl*/, int level, const char* fmt, va_list vl) {
    if (level > av_log_get_level()) return;

    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, vl);

    // Strip trailing newline FFmpeg appends
    int len = static_cast<int>(strlen(buf));
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        buf[--len] = '\0';

    if (len == 0) return;

    if      (level <= AV_LOG_FATAL)   { LOG_ERROR("[ffmpeg] {}", buf); }
    else if (level <= AV_LOG_WARNING) { LOG_WARN ("[ffmpeg] {}", buf); }
    else if (level <= AV_LOG_INFO)    { LOG_INFO ("[ffmpeg] {}", buf); }
    else                              { LOG_DEBUG("[ffmpeg] {}", buf); }
}

void configureFfmpegLogLevelOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        av_log_set_level(AV_LOG_WARNING);   // safe default; overridden by setFfmpegLogLevel()
        av_log_set_callback(ffmpegLogCallback);
    });
}

static int parseFfmpegLogLevel(const std::string& s) {
    if (s == "quiet")   return AV_LOG_QUIET;
    if (s == "panic")   return AV_LOG_PANIC;
    if (s == "fatal")   return AV_LOG_FATAL;
    if (s == "error")   return AV_LOG_ERROR;
    if (s == "warning") return AV_LOG_WARNING;
    if (s == "info")    return AV_LOG_INFO;
    if (s == "verbose") return AV_LOG_VERBOSE;
    if (s == "debug")   return AV_LOG_DEBUG;
    if (s == "trace")   return AV_LOG_TRACE;
    LOG_WARN("[ffmpeg] unknown log level '{}', defaulting to 'warning'", s);
    return AV_LOG_WARNING;
}
} // namespace

// ─── Free functions ───────────────────────────────────────────────────────────

int computeSampleInterval(double stream_fps, int sample_fps) {
    if (stream_fps <= 0.0) stream_fps = 25.0;
    return std::max(1, static_cast<int>(std::round(stream_fps / std::max(1, sample_fps))));
}

bool SamplingParams::shouldEmit(std::optional<double> pts, uint64_t frame_seq) {
    if (mode == SamplingMode::TimeBased) {
        const double min_gap = 1.0 / std::max(1, sample_fps);
        const double now = nowEpoch();
        if (last_emit_pts < 0.0 || (now - last_emit_pts) >= min_gap) {
            last_emit_pts = now;
            return true;
        }
        return false;
    }
    // FrameCount mode
    return (frame_seq % static_cast<uint64_t>(sample_interval)) == 0;
}

void setFfmpegLogLevel(const std::string& level) {
    configureFfmpegLogLevelOnce();   // ensure callback is registered
    av_log_set_level(parseFfmpegLogLevel(level));
}

AVPixelFormat FFmpegDecoder::getHWFormat(AVCodecContext* /*ctx*/, const AVPixelFormat* pix_fmts) {
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) return AV_PIX_FMT_CUDA;
    }
    return AV_PIX_FMT_NONE;
}

FFmpegDecoder::FFmpegDecoder() {
    configureFfmpegLogLevelOnce();
}

FFmpegDecoder::~FFmpegDecoder() {
    stop();
}

void FFmpegDecoder::start(const StreamConfig& cfg, FrameCallback cb) {
    if (running_.load()) return;
    stream_id_      = cfg.id;
    use_hwdec_      = cfg.use_hwdec;
    cuda_device_id_ = cfg.cuda_device_id;
    stop_flag_      = false;
    thread_     = std::thread(&FFmpegDecoder::decodeLoop, this, cfg, std::move(cb));
}

void FFmpegDecoder::stop() {
    stop_flag_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
    running_.store(false);
}

// ─── internal ────────────────────────────────────────────────────────────────

bool FFmpegDecoder::openStream(const StreamConfig& cfg) {
    AVDictionary* opts = nullptr;
    if (cfg.source_type == SourceType::RTSP) {
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        // stimeout covers the RTSP control-channel socket; aligned with open_timeout_us_
        av_dict_set(&opts, "stimeout", std::to_string(open_timeout_us_).c_str(), 0);
    }
    av_dict_set(&opts, "analyzeduration", "1000000", 0);
    av_dict_set(&opts, "probesize", "1000000", 0);

    // lavfi is an input device; register devices so av_find_input_format("lavfi") works.
    // avdevice_register_all() is idempotent.
    static std::once_flag avdevice_init;
    std::call_once(avdevice_init, avdevice_register_all);

    // "lavfi:<filtergraph>" URLs require the lavfi input device with the filtergraph
    // string passed as the filename (not the full "lavfi:<graph>" URL).
    // avformat_open_input takes AVInputFormat* in FFmpeg <5 and const AVInputFormat* in FFmpeg >=5.
    // Use the non-const pointer here to satisfy both.
    AVInputFormat* input_fmt = nullptr;
    std::string open_url = cfg.url;
    static constexpr std::string_view kLavfiPrefix = "lavfi:";
    if (cfg.url.compare(0, kLavfiPrefix.size(), kLavfiPrefix) == 0) {
        // Cast away const: pre-FFmpeg-5 avformat_open_input takes AVInputFormat* (non-const).
        input_fmt = const_cast<AVInputFormat*>(av_find_input_format("lavfi"));
        open_url  = cfg.url.substr(kLavfiPrefix.size());
    }

    // Pre-allocate so we can attach the interrupt callback before the blocking open.
    fmt_ctx_ = avformat_alloc_context();
    fmt_ctx_->interrupt_callback = {InterruptCtx::check, &interrupt_ctx_};

    interrupt_ctx_.arm(open_timeout_us_);
    int open_ret = avformat_open_input(&fmt_ctx_, open_url.c_str(), input_fmt, &opts);
    interrupt_ctx_.disarm();
    av_dict_free(&opts);

    if (open_ret != 0) {
        // avformat_open_input frees fmt_ctx_ on failure and sets it to nullptr.
        LOG_ERROR("[{}] avformat_open_input failed: {}", stream_id_, cfg.url);
        return false;
    }

    interrupt_ctx_.arm(open_timeout_us_);
    int info_ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    interrupt_ctx_.disarm();

    if (info_ret < 0) {
        LOG_ERROR("[{}] avformat_find_stream_info failed", stream_id_);
        closeStream();
        return false;
    }

    video_stream_idx_ = av_find_best_stream(
        fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        LOG_ERROR("[{}] no video stream found", stream_id_);
        closeStream();
        return false;
    }

    AVStream*       vs    = fmt_ctx_->streams[video_stream_idx_];
    const AVCodec*  codec = nullptr;

    if (use_hwdec_) {
        char decoder_name[64];
        const char* codec_name = avcodec_get_name(vs->codecpar->codec_id);
        snprintf(decoder_name, sizeof(decoder_name), "%s_cuvid", codec_name);
        codec = avcodec_find_decoder_by_name(decoder_name);
        if (!codec) {
            LOG_WARN("[{}] cuvid decoder '{}' not found, falling back to SW",
                     stream_id_, decoder_name);
            use_hwdec_ = false;
        }
    }

    if (!codec) {
        codec = avcodec_find_decoder(vs->codecpar->codec_id);
    }
    if (!codec) {
        LOG_ERROR("[{}] no decoder for codec_id {}", stream_id_, static_cast<int>(vs->codecpar->codec_id));
        closeStream();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("[{}] avcodec_alloc_context3 failed", stream_id_);
        closeStream();
        return false;
    }

    if (avcodec_parameters_to_context(codec_ctx_, vs->codecpar) < 0) {
        LOG_ERROR("[{}] avcodec_parameters_to_context failed", stream_id_);
        closeStream();
        return false;
    }

    if (use_hwdec_) {
        const std::string dev_str = std::to_string(cuda_device_id_);
        if (av_hwdevice_ctx_create(&hw_device_ctx_,
                                   AV_HWDEVICE_TYPE_CUDA,
                                   dev_str.c_str(), nullptr, 0) < 0) {
            LOG_WARN("[{}] av_hwdevice_ctx_create failed on device {}, falling back to SW",
                     stream_id_, cuda_device_id_);
            use_hwdec_ = false;
        } else {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            codec_ctx_->get_format    = &FFmpegDecoder::getHWFormat;
        }
    } else {
        codec_ctx_->thread_count = 2;
    }

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        LOG_ERROR("[{}] avcodec_open2 failed", stream_id_);
        closeStream();
        return false;
    }

    if (!use_hwdec_) {
        sws_ctx_ = sws_getContext(
            codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
            codec_ctx_->width, codec_ctx_->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }

    LOG_INFO("[{}] stream opened {}x{} hwdec={}",
             stream_id_, codec_ctx_->width, codec_ctx_->height, use_hwdec_);
    return true;
}

void FFmpegDecoder::closeStream() {
    if (sws_ctx_)        { sws_freeContext(sws_ctx_);       sws_ctx_ = nullptr; }
    if (hw_device_ctx_)  { av_buffer_unref(&hw_device_ctx_); }
    if (codec_ctx_)      { avcodec_free_context(&codec_ctx_); }
    if (fmt_ctx_)        { avformat_close_input(&fmt_ctx_); }
    video_stream_idx_ = -1;
}

ReadExitReason FFmpegDecoder::readAndDecode(FrameCallback& cb, SamplingParams& params) {
    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();
    ReadExitReason exit_reason = ReadExitReason::FramesOk;

    while (!stop_flag_.load()) {
        interrupt_ctx_.arm(read_timeout_us_);
        int ret = av_read_frame(fmt_ctx_, pkt);
        interrupt_ctx_.disarm();

        if (ret == AVERROR_EXIT) {
            LOG_WARN("[{}] av_read_frame timed out ({}ms), triggering reconnect",
                     stream_id_, read_timeout_us_ / 1000);
            exit_reason = ReadExitReason::Error;
            break;
        }
        if (ret == AVERROR(EAGAIN)) { av_usleep(1000); continue; }
        if (ret == AVERROR_EOF) {
            exit_reason = ReadExitReason::EndOfFile;
            break;
        }
        if (ret < 0) {
            exit_reason = ReadExitReason::Error;
            break;
        }

        if (pkt->stream_index == video_stream_idx_) {
            if (avcodec_send_packet(codec_ctx_, pkt) == 0) {
                while (avcodec_receive_frame(codec_ctx_, frame) == 0) {
                    const uint64_t seq = frame_seq_++;

                    std::optional<double> frame_pts;
                    if (frame->pts != AV_NOPTS_VALUE) {
                        AVStream* vs = fmt_ctx_->streams[video_stream_idx_];
                        frame_pts = frame->pts * av_q2d(vs->time_base);
                    }

                    if (params.shouldEmit(frame_pts, seq)) {
                        auto convert_start = std::chrono::steady_clock::now();

                        Frame f;
                        f.meta.stream_id   = stream_id_;
                        f.meta.capture_ts  = nowEpoch();
                        f.meta.capture_mono_ns = nowSteadyNs();
                        f.meta.frame_seq   = seq;
                        f.meta.orig_width  = codec_ctx_->width;
                        f.meta.orig_height = codec_ctx_->height;

                        if (use_hwdec_ && frame->format == AV_PIX_FMT_CUDA) {
                            // Keep the NVDEC NV12 surface alive by holding an
                            // av_frame_ref until the GpuBuffer is destroyed.
                            AVFrame* hw_ref = av_frame_alloc();
                            if (av_frame_ref(hw_ref, frame) < 0) {
                                av_frame_free(&hw_ref);
                                LOG_WARN("[{}] av_frame_ref failed, dropping frame",
                                         stream_id_);
                                Metrics::get().incFramesDropped(stream_id_);
                                av_frame_unref(frame);
                                continue;
                            }

                            f.is_gpu = true;
                            f.gpu_buf.y_data        = static_cast<void*>(hw_ref->data[0]);
                            f.gpu_buf.uv_data       = static_cast<void*>(hw_ref->data[1]);
                            f.gpu_buf.width         = codec_ctx_->width;
                            f.gpu_buf.height        = codec_ctx_->height;
                            f.gpu_buf.cuda_stream   = nullptr;
                            f.gpu_buf.cuda_device_id = cuda_device_id_;
                            // shared_ptr deleter releases the hw buffer
                            f.gpu_buf.frame_ref = std::shared_ptr<void>(
                                hw_ref,
                                [](void* p) {
                                    AVFrame* fr = static_cast<AVFrame*>(p);
                                    av_frame_unref(fr);
                                    av_frame_free(&fr);
                                });
                        } else {
                            cv::Mat bgr(codec_ctx_->height, codec_ctx_->width, CV_8UC3);
                            uint8_t* dst[1]    = { bgr.data };
                            int      stride[1] = { static_cast<int>(bgr.step) };
                            sws_scale(sws_ctx_, frame->data, frame->linesize,
                                      0, codec_ctx_->height, dst, stride);
                            f.image  = std::move(bgr);
                            f.is_gpu = false;
                        }

                        const double convert_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - convert_start).count();

                        Metrics::get().incFramesDecoded(stream_id_);
                        StreamHealthRegistry::get().onFrameDecoded(stream_id_, f.meta.capture_ts);
                        if (f.meta.frame_seq % 30 == 0) {
                            LOG_DEBUG("[{}] frame_seq={} ts={:.3f} {}x{} convert_ms={:.1f}",
                                      stream_id_, f.meta.frame_seq, f.meta.capture_ts,
                                      f.meta.orig_width, f.meta.orig_height, convert_ms);
                        }
                        cb(std::move(f));
                    }
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    return exit_reason;
}

// Compute exponential backoff delay in milliseconds.
// delay = min(base * 2^failures, ceiling), with ±10% jitter.
int64_t FFmpegDecoder::backoffDelayMs(int base_ms, int ceiling_ms, uint32_t failures) {
    int64_t delay = static_cast<int64_t>(base_ms) * (1LL << std::min(failures, 6u));
    delay = std::min(delay, static_cast<int64_t>(ceiling_ms));
    // ±10% jitter to avoid thundering herd
    int64_t jitter = (delay / 10);
    if (jitter > 0) {
        std::uniform_int_distribution<int64_t> dist(-jitter, jitter);
        delay += dist(rng_);
    }
    return std::max(delay, static_cast<int64_t>(base_ms));
}

void FFmpegDecoder::decodeLoop(StreamConfig cfg, FrameCallback cb) {
    running_.store(true);
    LOG_INFO("[{}] decode thread started: {} hwdec={}", cfg.id, cfg.url, cfg.use_hwdec);

    read_timeout_us_ = static_cast<int64_t>(cfg.read_timeout_ms) * 1000LL;
    open_timeout_us_ = static_cast<int64_t>(cfg.open_timeout_ms) * 1000LL;

    auto& reg = StreamHealthRegistry::get();
    reg.onStreamAdded(cfg.id, cfg.degraded_threshold, cfg.max_reconnect_attempts);

    while (!stop_flag_.load()) {
        if (!openStream(cfg)) {
            // File sources: a failed open is terminal — don't retry.
            if (cfg.source_type == SourceType::File) {
                LOG_ERROR("[{}] failed to open file: {}", cfg.id, cfg.url);
                break;
            }
            reg.onReconnectFailed(cfg.id);
            const auto h = reg.getHealth(cfg.id);
            if (h.state == StreamState::FAILED) {
                ControlEventBus::get().emit(
                    ControlEventType::StreamFailedTerminal,
                    cfg.id,
                    h.state,
                    nowEpoch(),
                    static_cast<int>(h.consecutive_failures),
                    cfg.max_reconnect_attempts,
                    "max reconnect attempts reached");
                LOG_ERROR("[{}] reached terminal reconnect failure (attempt {}), stopping decoder thread",
                          cfg.id, h.consecutive_failures);
                break;
            }
            uint32_t failures = h.consecutive_failures;
            int64_t delay = backoffDelayMs(cfg.reconnect_delay_ms,
                                           cfg.max_reconnect_delay_ms, failures);
            LOG_WARN("[{}] open failed (attempt {}), retry in {}ms",
                     cfg.id, failures, delay);
            if (waitForOrStop(stop_flag_, std::chrono::milliseconds(delay))) {
                break;
            }
            continue;
        }

        // Distinguish first open (CONNECTING) from reconnect (RECONNECTING/DEGRADED)
        {
            StreamState cur = reg.getHealth(cfg.id).state;
            if (cur == StreamState::CONNECTING)
                reg.onStreamOpened(cfg.id);
            else {
                reg.onReconnectSucceeded(cfg.id);
                const auto hh = reg.getHealth(cfg.id);
                ControlEventBus::get().emit(
                    ControlEventType::StreamRecovered,
                    cfg.id,
                    hh.state,
                    nowEpoch(),
                    static_cast<int>(hh.consecutive_failures),
                    cfg.max_reconnect_attempts,
                    "");
            }
        }

        // Build SamplingParams from actual stream fps (only valid after openStream()).
        frame_seq_ = 0;
        SamplingParams params;
        params.mode      = cfg.sampling_mode;
        params.sample_fps = cfg.sample_fps;
        {
            AVStream* vs = fmt_ctx_->streams[video_stream_idx_];
            params.sample_interval = computeSampleInterval(av_q2d(vs->avg_frame_rate), cfg.sample_fps);
        }
        LOG_INFO("[{}] sampling mode={} sample_fps={} interval={}",
                 cfg.id,
                 params.mode == SamplingMode::TimeBased ? "time_based" : "frame_count",
                 params.sample_fps, params.sample_interval);

        auto reason = readAndDecode(cb, params);
        closeStream();

        // File sources: EOF is a clean terminal; optionally loop by re-opening.
        if (cfg.source_type == SourceType::File) {
            if (reason == ReadExitReason::EndOfFile && cfg.loop_file && !stop_flag_.load()) {
                LOG_INFO("[{}] EOF, looping file", cfg.id);
                continue;  // outer while re-opens the file from the start
            }
            LOG_INFO("[{}] file playback complete", cfg.id);
            break;  // clean exit — no StreamDropped event
        }

        // RTSP: treat any exit (including timeout/error) as a drop → reconnect.
        if (!stop_flag_.load()) {
            reg.onStreamDropped(cfg.id);
            const auto h = reg.getHealth(cfg.id);
            ControlEventBus::get().emit(
                ControlEventType::StreamDropped,
                cfg.id,
                h.state,
                nowEpoch(),
                static_cast<int>(h.consecutive_failures),
                cfg.max_reconnect_attempts,
                "stream dropped");
            uint32_t failures = reg.getHealth(cfg.id).consecutive_failures;
            int64_t delay = backoffDelayMs(cfg.reconnect_delay_ms,
                                           cfg.max_reconnect_delay_ms, failures);
            LOG_WARN("[{}] stream dropped, reconnecting in {}ms", cfg.id, delay);
            if (waitForOrStop(stop_flag_, std::chrono::milliseconds(delay))) {
                break;
            }
        }
    }

    reg.onStreamRemoved(cfg.id);
    LOG_INFO("[{}] decode thread stopped", cfg.id);
    running_.store(false);
}

} // namespace infer
