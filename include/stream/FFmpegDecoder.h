#pragma once

#include "stream/IStreamDecoder.h"
#include <thread>
#include <atomic>
#include <string>

// Forward-declare FFmpeg types to keep this header clean
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace infer {

// RTSP decoder backed by FFmpeg.
// Phase 1: software decode (libavcodec).
// Phase 4: switch to NVDEC via hwaccel=cuda.
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
    bool openStream(const std::string& url);
    void closeStream();
    bool readAndDecode(FrameCallback& cb, int sample_interval);

    std::string          stream_id_;
    std::thread          thread_;
    std::atomic<bool>    running_{false};
    std::atomic<bool>    stop_flag_{false};

    // FFmpeg state (owned by decodeLoop thread)
    AVFormatContext* fmt_ctx_{nullptr};
    AVCodecContext*  codec_ctx_{nullptr};
    SwsContext*      sws_ctx_{nullptr};
    int              video_stream_idx_{-1};
    uint64_t         frame_seq_{0};
};

} // namespace infer
