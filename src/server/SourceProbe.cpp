#include "server/SourceProbe.h"
#include "server/BoundedChild.h"
#include "stream/InterruptCtx.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace infer {
namespace {

using namespace std::chrono;
constexpr auto kMaxProbeTime = seconds(15);

struct Input {
    AVFormatContext* context{avformat_alloc_context()};
    ~Input() { avformat_close_input(&context); }
};

struct CodecFree {
    void operator()(AVCodecContext* context) const { avcodec_free_context(&context); }
    void operator()(AVFrame* frame) const { av_frame_free(&frame); }
    void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

bool timedOut(int result, InterruptCtx& interrupt) {
    return result == AVERROR_EXIT || result == AVERROR(ETIMEDOUT) || InterruptCtx::check(&interrupt);
}

SourceProbeCode decodeFirstFrame(AVFormatContext* input, int video_index, InterruptCtx& interrupt) {
    const auto* parameters = input->streams[video_index]->codecpar;
    const auto* codec = avcodec_find_decoder(parameters->codec_id);
    if (!codec) return SourceProbeCode::DecodeFailed;
    std::unique_ptr<AVCodecContext, CodecFree> decoder(avcodec_alloc_context3(codec));
    std::unique_ptr<AVPacket, CodecFree> packet(av_packet_alloc());
    std::unique_ptr<AVFrame, CodecFree> frame(av_frame_alloc());
    if (!decoder || !packet || !frame) return SourceProbeCode::Unavailable;
    if (avcodec_parameters_to_context(decoder.get(), parameters) < 0) return SourceProbeCode::DecodeFailed;
    decoder->thread_count = 1;
    if (avcodec_open2(decoder.get(), codec, nullptr) < 0) return SourceProbeCode::DecodeFailed;

    bool decode_failed = false;
    while (!InterruptCtx::check(&interrupt)) {
        const int result = av_read_frame(input, packet.get());
        if (result == AVERROR(EAGAIN)) {
            std::this_thread::sleep_for(milliseconds(1));
            continue;
        }
        const bool end = result == AVERROR_EOF;
        if (result < 0 && !end) {
            if (!timedOut(result, interrupt)) {
                return decode_failed ? SourceProbeCode::DecodeFailed : SourceProbeCode::StreamFailed;
            }
            break;
        }
        if (end || packet->stream_index == video_index) {
            const int sent = avcodec_send_packet(decoder.get(), end ? nullptr : packet.get());
            if (sent < 0 && sent != AVERROR_EOF) decode_failed = true;
            while (!InterruptCtx::check(&interrupt)) {
                const int received = avcodec_receive_frame(decoder.get(), frame.get());
                if (received < 0) {
                    if (received != AVERROR(EAGAIN) && received != AVERROR_EOF) decode_failed = true;
                    break;
                }
                if (frame->width > 0 && frame->height > 0 &&
                    !(frame->flags & AV_FRAME_FLAG_CORRUPT)) {
                    return SourceProbeCode::Passed;
                }
                decode_failed = true;
                av_frame_unref(frame.get());
            }
        }
        av_packet_unref(packet.get());
        if (end) return decode_failed ? SourceProbeCode::DecodeFailed : SourceProbeCode::StreamFailed;
    }
    return decode_failed ? SourceProbeCode::DecodeFailed : SourceProbeCode::FirstFrameTimeout;
}

SourceProbeCode probe(const std::string& url, int open_ms, int read_ms) {
    if (url.compare(0, 7, "rtsp://") != 0 || url.find('\0') != std::string::npos) {
        return SourceProbeCode::ConnectionFailed;
    }
    av_log_set_level(AV_LOG_QUIET);
    const auto* format = av_find_input_format("rtsp");
    if (!format) return SourceProbeCode::Unavailable;
    InterruptCtx interrupt;
    Input input;
    if (!input.context) return SourceProbeCode::Unavailable;
    input.context->interrupt_callback = {InterruptCtx::check, &interrupt};

    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "protocol_whitelist", "tcp,rtp", 0);
    interrupt.arm(static_cast<int64_t>(std::clamp(open_ms, 1, 15000)) * 1000);
    // FFmpeg before 5 expects a non-const input format pointer.
    const int opened = avformat_open_input(&input.context, url.c_str(),
        const_cast<AVInputFormat*>(format), &options);
    av_dict_free(&options);
    if (opened < 0) {
        if (opened == AVERROR_HTTP_UNAUTHORIZED || opened == AVERROR_HTTP_FORBIDDEN) {
            return SourceProbeCode::AuthenticationFailed;
        }
        return timedOut(opened, interrupt) ? SourceProbeCode::ConnectionTimeout
                                           : SourceProbeCode::ConnectionFailed;
    }

    // RTSP supplies the track list and codec parameters in SDP during open.
    const int video_index = av_find_best_stream(input.context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_index < 0) return SourceProbeCode::NoVideo;
    interrupt.arm(static_cast<int64_t>(std::clamp(read_ms, 1, 15000)) * 1000);
    return decodeFirstFrame(input.context, video_index, interrupt);
}

struct SpawnActions {
    posix_spawn_file_actions_t actions;
    int error{posix_spawn_file_actions_init(&actions)};
    ~SpawnActions() {
        if (error == 0) posix_spawn_file_actions_destroy(&actions);
    }
};

} // namespace

int runSourceProbe() {
    try {
        nlohmann::json request;
        std::cin >> request;
        return static_cast<int>(probe(request.at("url").get<std::string>(),
            request.at("open_timeout_ms").get<int>(), request.at("read_timeout_ms").get<int>()));
    } catch (...) {
        return static_cast<int>(SourceProbeCode::Unavailable);
    }
}

SourceProbeCode probeSourceInChild(const std::string& executable, const PipelineSourceConfig& source) {
    const auto request = nlohmann::json{{"url", source.url}, {"open_timeout_ms", source.open_timeout_ms},
                                        {"read_timeout_ms", source.read_timeout_ms}}.dump();
    // Pass credentials through an anonymous temporary file, outside the child command line.
    using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
    File input(std::tmpfile(), &std::fclose);
    if (!input) return SourceProbeCode::Unavailable;
    const int fd = fileno(input.get());
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) == -1 ||
        std::fwrite(request.data(), 1, request.size(), input.get()) != request.size() ||
        std::fseek(input.get(), 0, SEEK_SET) != 0) {
        return SourceProbeCode::Unavailable;
    }
    SpawnActions file_actions;
    if (file_actions.error ||
        posix_spawn_file_actions_adddup2(&file_actions.actions, fd, STDIN_FILENO) != 0) {
        return SourceProbeCode::Unavailable;
    }
    auto program = executable;
    char flag[] = "--probe-source";
    char* argv[] = {program.data(), flag, nullptr};
    const auto status = runBoundedChild(argv, environ, kMaxProbeTime, &file_actions.actions);
    if (!status) return SourceProbeCode::ProbeTimeout;
    if (!WIFEXITED(*status)) return SourceProbeCode::Unavailable;

    const int code = WEXITSTATUS(*status);
    if (code == 0) return SourceProbeCode::Passed;
    if (code >= static_cast<int>(SourceProbeCode::ConnectionFailed) &&
        code <= static_cast<int>(SourceProbeCode::StreamFailed)) {
        return static_cast<SourceProbeCode>(code);
    }
    return SourceProbeCode::Unavailable;
}

} // namespace infer
