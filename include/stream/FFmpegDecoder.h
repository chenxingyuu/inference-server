#pragma once

#include "common/Config.h"
#include "stream/IStreamDecoder.h"
#include "stream/InterruptCtx.h"
#include <optional>
#include <thread>
#include <atomic>
#include <string>
#include <random>

// Forward-declare FFmpeg types to keep this header clean
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
struct AVBufferRef;
// AVPixelFormat is an enum; forward-declare via the C header enum trick
extern "C" {
#include <libavutil/pixfmt.h>
}

namespace infer {

// Compute frame-count sample interval from actual and target fps.
// Falls back to 25 fps if stream_fps <= 0. Result is always >= 1.
int computeSampleInterval(double stream_fps, int sample_fps);

// Per-reconnect sampling state passed into readAndDecode().
struct SamplingParams {
    SamplingMode mode{SamplingMode::FrameCount};
    int          sample_fps{5};
    int          sample_interval{1};   // used in FrameCount mode
    double       last_emit_pts{-1.0};  // TimeBased: wall-clock epoch seconds, -1 = unset

    // Returns true if the current frame should be emitted.
    // pts: unused in TimeBased mode (wall clock used instead).
    // frame_seq: 0-based decoded-frame count since last reconnect.
    bool shouldEmit(std::optional<double> pts, uint64_t frame_seq);
};

// Set FFmpeg's global log threshold at runtime.
// Valid values: "quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug", "trace".
// Typically called once after loading AppConfig. Env var FFMPEG_LOG_LEVEL takes precedence.
void setFfmpegLogLevel(const std::string& level);

// CPU software decode: libavcodec thread_count (see server.ffmpeg_decode_threads in YAML).
// Call once after loadConfig. 0 = do not set (libavcodec default).
void setFfmpegDecodeThreads(int threads);

// Reason that readAndDecode() returned.
enum class ReadExitReason {
    FramesOk,   // stopped by stop_flag_
    EndOfFile,  // clean end-of-file (AVERROR_EOF)
    Error       // read error or timeout
};

// Decoder backed by FFmpeg supporting both RTSP streams and local files.
// Software path: libavcodec → cv::Mat BGR
// Hardware path (use_hwdec=true): NVDEC → NV12 GPU buffer (GpuBuffer)
class FFmpegDecoder : public IStreamDecoder {
public:
    FFmpegDecoder();
    ~FFmpegDecoder() override;

    void start(const StreamConfig& cfg, FrameCallback cb) override;
    void stop() override;
    bool running() const override { return running_.load(); }
    const std::string& streamId() const override { return stream_id_; }

private:
    void decodeLoop(StreamConfig cfg, FrameCallback cb);
    bool openStream(const StreamConfig& cfg);
    void closeStream();
    ReadExitReason readAndDecode(FrameCallback& cb, SamplingParams& params);
    int64_t backoffDelayMs(int base_ms, int ceiling_ms, uint32_t failures);

    // Called by FFmpeg to select the hw pixel format
    static AVPixelFormat getHWFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

    std::string          stream_id_;
    std::thread          thread_;
    std::atomic<bool>    running_{false};
    std::atomic<bool>    stop_flag_{false};

    // FFmpeg state (owned by decodeLoop thread)
    AVFormatContext* fmt_ctx_{nullptr};
    AVCodecContext*  codec_ctx_{nullptr};
    SwsContext*      sws_ctx_{nullptr};
    AVBufferRef*     hw_device_ctx_{nullptr};
    int              video_stream_idx_{-1};
    uint64_t         frame_seq_{0};
    bool             use_hwdec_{false};
    int              cuda_device_id_{0};

    // Interrupt callback used to time-bound av_read_frame / avformat_open_input.
    // Owned by decodeLoop thread; written before blocking calls, cleared after.
    InterruptCtx     interrupt_ctx_;
    int64_t          read_timeout_us_{5'000'000LL};   // default 5 s
    int64_t          open_timeout_us_{10'000'000LL};  // default 10 s
    std::mt19937     rng_{std::random_device{}()};
};

} // namespace infer
