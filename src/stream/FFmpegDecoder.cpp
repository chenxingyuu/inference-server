#include "stream/FFmpegDecoder.h"
#include "stream/StreamHealthRegistry.h"
#include "common/Logger.h"
#include "metrics/Metrics.h"
#include "publisher/ControlEventBus.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
}

#include <opencv2/imgproc.hpp>
#include <chrono>
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

void configureFfmpegLogLevelOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        // Reduce noisy decoder warnings (e.g. SEI truncation) and keep errors.
        av_log_set_level(AV_LOG_ERROR);
    });
}
} // namespace

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
    stream_id_  = cfg.id;
    use_hwdec_  = cfg.use_hwdec;
    stop_flag_  = false;
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
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "analyzeduration", "1000000", 0);
    av_dict_set(&opts, "probesize", "1000000", 0);

    fmt_ctx_ = nullptr;
    if (avformat_open_input(&fmt_ctx_, cfg.url.c_str(), nullptr, &opts) != 0) {
        av_dict_free(&opts);
        LOG_ERROR("[{}] avformat_open_input failed: {}", stream_id_, cfg.url);
        return false;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
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
        if (av_hwdevice_ctx_create(&hw_device_ctx_,
                                   AV_HWDEVICE_TYPE_CUDA,
                                   nullptr, nullptr, 0) < 0) {
            LOG_WARN("[{}] av_hwdevice_ctx_create failed, falling back to SW",
                     stream_id_);
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

bool FFmpegDecoder::readAndDecode(FrameCallback& cb, int sample_interval) {
    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();
    bool ok = false;

    while (!stop_flag_.load()) {
        int ret = av_read_frame(fmt_ctx_, pkt);
        if (ret == AVERROR(EAGAIN)) { av_usleep(1000); continue; }
        if (ret < 0) { break; }

        if (pkt->stream_index == video_stream_idx_) {
            if (avcodec_send_packet(codec_ctx_, pkt) == 0) {
                while (avcodec_receive_frame(codec_ctx_, frame) == 0) {
                    if ((frame_seq_++ % sample_interval) == 0) {
                        Frame f;
                        f.meta.stream_id   = stream_id_;
                        f.meta.capture_ts  = nowEpoch();
                        f.meta.capture_mono_ns = nowSteadyNs();
                        f.meta.frame_seq   = frame_seq_;
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
                            f.gpu_buf.y_data  = static_cast<void*>(hw_ref->data[0]);
                            f.gpu_buf.uv_data = static_cast<void*>(hw_ref->data[1]);
                            f.gpu_buf.width   = codec_ctx_->width;
                            f.gpu_buf.height  = codec_ctx_->height;
                            f.gpu_buf.cuda_stream = nullptr;
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

                        Metrics::get().incFramesDecoded(stream_id_);
                        StreamHealthRegistry::get().onFrameDecoded(stream_id_, f.meta.capture_ts);
                        cb(std::move(f));
                        ok = true;
                    }
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    return ok;
}

// Compute exponential backoff delay in milliseconds.
// delay = min(base * 2^failures, ceiling), with ±10% jitter.
static int64_t backoffDelayMs(int base_ms, int ceiling_ms, uint32_t failures) {
    int64_t delay = static_cast<int64_t>(base_ms) * (1LL << std::min(failures, 6u));
    delay = std::min(delay, static_cast<int64_t>(ceiling_ms));
    // ±10% jitter to avoid thundering herd
    int64_t jitter = (delay / 10);
    if (jitter > 0)
        delay += (static_cast<int64_t>(rand()) % (jitter * 2 + 1)) - jitter;
    return std::max(delay, static_cast<int64_t>(base_ms));
}

void FFmpegDecoder::decodeLoop(StreamConfig cfg, FrameCallback cb) {
    running_.store(true);
    LOG_INFO("[{}] decode thread started: {} hwdec={}", cfg.id, cfg.url, cfg.use_hwdec);

    auto& reg = StreamHealthRegistry::get();
    reg.onStreamAdded(cfg.id, cfg.degraded_threshold, cfg.max_reconnect_attempts);

    const int sample_interval = std::max(1, 25 / std::max(1, cfg.sample_fps));

    while (!stop_flag_.load()) {
        if (!openStream(cfg)) {
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
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
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

        readAndDecode(cb, sample_interval);
        closeStream();

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
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }

    reg.onStreamRemoved(cfg.id);
    LOG_INFO("[{}] decode thread stopped", cfg.id);
    running_.store(false);
}

} // namespace infer
