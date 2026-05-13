#ifdef BUILD_ASCEND_BACKEND

#include "pipeline/stages/AscendVencFfmpegMuxWriter.h"

#include "infer/AscendProcessRuntime.h"
#include "infer/BgrToNv12.h"
#include "common/Logger.h"

#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <thread>

namespace infer {

namespace {

std::string buildOutputUrl(const std::string& url, const std::string& protocol) {
    if (protocol != "rtsp") return url;
    if (url.find("rtsp_transport=") != std::string::npos) return url;
    if (url.find('?') == std::string::npos) return url + "?rtsp_transport=tcp";
    return url + "&rtsp_transport=tcp";
}

std::string shellQuote(const std::string& input) {
    std::string out = "'";
    for (const char c : input) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}

/// Log at WARN as well as ERROR: some deployments only surface WARN from `DrawAndStreamStage` peers.
void logVencOpenAclFailure(const char* step, aclError rc) {
    const int code = static_cast<int>(rc);
    LOG_ERROR("AscendVencFfmpegMuxWriter: {} failed ({})", step, code);
    LOG_WARN("AscendVencFfmpegMuxWriter(open): {} failed ({})", step, code);
    // CANN: 507018 == ACL_ERROR_RT_AICPU_EXCEPTION (media/DVPP sidecar); often seen when VENC params or
    // resolution exceed silicon limits, or DVPP+VENC contention under load.
    if (code == 507018) {
        LOG_WARN(
            "AscendVencFfmpegMuxWriter(open): 507018=AICPU exception on VENC path — try encoder=ffmpeg_x264, "
            "or reduce frame size (very wide e.g. 7680+padded H may exceed VENC spec on 300I Pro)");
    }
    if (code == 100000) {
        LOG_WARN(
            "AscendVencFfmpegMuxWriter(open): 100000=ACL_ERROR_INVALID_PARAM — often output bufSize vs "
            "pic_w×pic_h mismatch on Atlas inference VENC");
    }
}

#define ACL_CHECK_OR_RETURN(call, retval)                                                                          \
    do {                                                                                                           \
        aclError _err = (call);                                                                                    \
        if (_err != ACL_SUCCESS) {                                                                                 \
            LOG_ERROR("AscendVencFfmpegMuxWriter: ACL error {} at {}:{}", static_cast<int>(_err), __FILE__,         \
                      __LINE__);                                                                                   \
            return (retval);                                                                                       \
        }                                                                                                          \
    } while (0)

} // namespace

AscendVencFfmpegMuxWriter::AscendVencFfmpegMuxWriter(int device_id) : device_id_(device_id) {}

AscendVencFfmpegMuxWriter::~AscendVencFfmpegMuxWriter() { close(); }

bool AscendVencFfmpegMuxWriter::isOpened() const { return opened_.load(std::memory_order_acquire); }

static void releaseRuntimeIf(bool& flag) {
    if (flag) {
        AscendProcessRuntime::release();
        flag = false;
    }
}

void AscendVencFfmpegMuxWriter::rollbackOpen() {
    closeFfmpegPipe();
    if (ctx_) {
        (void)aclrtSetCurrentContext(ctx_);
    }
    destroyEncoder();
    if (stream_) {
        (void)aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (ctx_) {
        (void)aclrtDestroyContext(ctx_);
        ctx_ = nullptr;
    }
    releaseRuntimeIf(runtime_acquired_);
    opened_.store(false, std::memory_order_release);
}

void AscendVencFfmpegMuxWriter::closeFfmpegPipe() {
    if (pipe_ != nullptr) {
        (void)std::fflush(pipe_);
        const int rc = pclose(pipe_);
        pipe_ = nullptr;
        if (rc != 0) {
            LOG_WARN("AscendVencFfmpegMuxWriter: ffmpeg mux exited with status {}", rc);
        }
    }
}

void AscendVencFfmpegMuxWriter::destroyEncoder() {
    if (channel_desc_) {
        if (venc_channel_created_) {
            (void)aclvencDestroyChannel(channel_desc_);
            venc_channel_created_ = false;
        }
        (void)aclvencDestroyChannelDesc(channel_desc_);
        channel_desc_ = nullptr;
    }
    if (venc_out_buf_) {
        (void)acldvppFree(venc_out_buf_);
        venc_out_buf_      = nullptr;
        venc_out_buf_size_ = 0;
    }
}

void AscendVencFfmpegMuxWriter::onVencCallback(acldvppPicDesc* input, acldvppStreamDesc* output, void* userdata) {
    auto* self = static_cast<AscendVencFfmpegMuxWriter*>(userdata);
    if (!self) return;

    if (output) {
        void*          data = acldvppGetStreamDescData(output);
        const uint32_t sz  = acldvppGetStreamDescSize(output);
        if (data && sz > 0 && self->pipe_) {
            std::vector<uint8_t> host_packet(static_cast<size_t>(sz));
            if (self->ctx_) {
                const aclError set_ctx_rc = aclrtSetCurrentContext(self->ctx_);
                if (set_ctx_rc != ACL_SUCCESS) {
                    LOG_WARN("AscendVencFfmpegMuxWriter: callback aclrtSetCurrentContext failed ({})",
                             static_cast<int>(set_ctx_rc));
                }
            }
            const aclError copy_rc = aclrtMemcpy(
                host_packet.data(),
                static_cast<size_t>(sz),
                data,
                static_cast<size_t>(sz),
                ACL_MEMCPY_DEVICE_TO_HOST);
            if (copy_rc != ACL_SUCCESS) {
                LOG_WARN("AscendVencFfmpegMuxWriter: callback D2H copy failed ({}) size={}",
                         static_cast<int>(copy_rc),
                         sz);
            } else {
            std::lock_guard<std::mutex> lk(self->pipe_mu_);
                const size_t wrote = std::fwrite(host_packet.data(), 1, static_cast<size_t>(sz), self->pipe_);
            if (wrote != static_cast<size_t>(sz)) {
                    const int e = errno;
                    LOG_WARN("AscendVencFfmpegMuxWriter: short fwrite to ffmpeg ({} / {}), errno={} ({})",
                             wrote,
                             sz,
                             e,
                             std::strerror(e));
            }
                (void)std::fflush(self->pipe_);
            }
        }
    }

    if (input) {
        void* in_data = acldvppGetPicDescData(input);
        if (in_data) {
            const aclError fr = acldvppFree(in_data);
            if (fr != ACL_SUCCESS) {
                LOG_WARN("AscendVencFfmpegMuxWriter: acldvppFree input failed ({})", static_cast<int>(fr));
            }
        }
        const aclError dr = acldvppDestroyPicDesc(input);
        if (dr != ACL_SUCCESS) {
            LOG_WARN("AscendVencFfmpegMuxWriter: acldvppDestroyPicDesc input failed ({})", static_cast<int>(dr));
        }
    }

    self->frame_done_.store(true, std::memory_order_release);
}

bool AscendVencFfmpegMuxWriter::open(const std::string& url,
                                     const std::string& protocol,
                                     double fps,
                                     int gop,
                                     int bitrate_kbps,
                                     int width,
                                     int height) {
    close();

    if (url.empty()) return false;
    for (unsigned char c : url) {
        if (c < 0x20 || c == 0x7f) return false;
    }

    // H.264 / VENC macroblock grid: both dimensions should be multiples of 16 (width was already
    // documented this way; height must not be only 2-aligned — e.g. 1144 fails, 1152 works).
    pic_w_ = static_cast<uint32_t>((static_cast<uint32_t>(width) + 15u) & ~15u);
    pic_h_ = static_cast<uint32_t>((static_cast<uint32_t>(height) + 15u) & ~15u);
    if (pic_w_ == 0 || pic_h_ == 0) return false;
    if (pic_w_ != static_cast<uint32_t>(width) || pic_h_ != static_cast<uint32_t>(height)) {
        LOG_INFO("AscendVencFfmpegMuxWriter: aligned frame {}x{} -> VENC {}x{}", width, height, pic_w_, pic_h_);
    }

    nv12_host_bytes_ = static_cast<size_t>(pic_w_) * static_cast<size_t>(pic_h_) * 3u / 2u;
    nv12_host_.assign(nv12_host_bytes_, 0);

    // Fork the ffmpeg muxer before any NPU/DVPP/VENC init. `popen` after `aclvencCreateChannel` has been
    // observed to fail on some Ascend stacks (parent holds active encode channel while fork+exec runs).
    const std::string output_url = buildOutputUrl(url, protocol);
    std::string       mux_format = "rtsp";
    if (protocol == "rtmp") mux_format = "flv";
    const int fps_i = std::max(1, static_cast<int>(std::lround(fps)));
    // Raw Annex-B H264 on stdin: give demuxer a nominal framerate + genpts; live FLV/RTMP needs
    // flvflags so ffmpeg does not bail before the first VENC NAL arrives.
    std::string out_extra;
    if (protocol == "rtmp") {
        out_extra = "-flvflags no_duration_filesize ";
    }
    const std::string cmd = "ffmpeg -hide_banner -loglevel error -nostats "
                            "-fflags +nobuffer+genpts "
                            "-framerate " + std::to_string(fps_i) + " "
                            "-f h264 -i - "
                            "-an -c:v copy "
                            "-f " + mux_format + " " + out_extra + shellQuote(output_url);
    pipe_ = popen(cmd.c_str(), "w");
    if (pipe_ == nullptr) {
        const int e = errno;
        LOG_ERROR("AscendVencFfmpegMuxWriter: popen ffmpeg failed errno={} ({})", e, std::strerror(e));
        LOG_WARN("AscendVencFfmpegMuxWriter(open): popen ffmpeg failed errno={} ({})", e, std::strerror(e));
        return false;
    }

    try {
        AscendProcessRuntime::acquire();
    } catch (const std::exception& e) {
        LOG_ERROR("AscendVencFfmpegMuxWriter: AscendProcessRuntime::acquire failed: {}", e.what());
        LOG_WARN("AscendVencFfmpegMuxWriter(open): AscendProcessRuntime::acquire failed: {}", e.what());
        closeFfmpegPipe();
        return false;
    }
    runtime_acquired_ = true;

    aclError rc = aclrtSetDevice(device_id_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclrtSetDevice", rc);
        rollbackOpen();
        return false;
    }
    rc = aclrtCreateContext(&ctx_, device_id_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclrtCreateContext", rc);
        rollbackOpen();
        return false;
    }
    rc = aclrtCreateStream(&stream_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclrtCreateStream", rc);
        rollbackOpen();
        return false;
    }
    // DVPP/VENC APIs allocate and bind to the active thread context (same pattern as
    // AscendBackend::infer / AscendBufferPool::reset). Without this, acldvppMalloc /
    // aclvencCreateChannel may fail or target the wrong context on CANN 6.
    rc = aclrtSetCurrentContext(ctx_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclrtSetCurrentContext", rc);
        rollbackOpen();
        return false;
    }

    // Output bitstream buffer: acldvppMalloc + aclvencSetChannelDescBufSize must agree. A fixed 4MiB
    // A fixed 4MiB floor exceeded the firmware-valid range vs pic_w×pic_h on 1080p (CANN 6
    // ACL_ERROR_INVALID_PARAM on aclvencSetChannelDescBufSize). Use a modest multiple of pic area.
    // (100000) on aclvencSetChannelDescBufSize (seen on 1920×1088).
    {
        const uint64_t pic_area   = static_cast<uint64_t>(pic_w_) * static_cast<uint64_t>(pic_h_);
        constexpr uint64_t kMin = static_cast<uint64_t>(512) * 1024;
        constexpr uint64_t kMax = static_cast<uint64_t>(32) * 1024 * 1024;
        // Explicit template args: GCC 9 in the CANN 6 image otherwise picks wrong std::min overload.
        uint64_t sz = std::max<uint64_t>(pic_area * static_cast<uint64_t>(3), kMin);
        sz            = std::min<uint64_t>(sz, kMax);
        venc_out_buf_size_ = static_cast<uint32_t>(sz);
    }
    rc = acldvppMalloc(&venc_out_buf_, static_cast<size_t>(venc_out_buf_size_));
    if (rc != ACL_SUCCESS || !venc_out_buf_) {
        if (rc == ACL_SUCCESS) {
            LOG_ERROR("AscendVencFfmpegMuxWriter: acldvppMalloc out buffer returned null");
            LOG_WARN("AscendVencFfmpegMuxWriter(open): acldvppMalloc out buffer returned null");
        } else {
            logVencOpenAclFailure("acldvppMalloc(out)", rc);
        }
        rollbackOpen();
        return false;
    }

    channel_desc_ = aclvencCreateChannelDesc();
    if (!channel_desc_) {
        LOG_ERROR("AscendVencFfmpegMuxWriter: aclvencCreateChannelDesc returned null");
        LOG_WARN("AscendVencFfmpegMuxWriter(open): aclvencCreateChannelDesc returned null");
        rollbackOpen();
        return false;
    }

    const uint32_t gop_u = gop > 0 ? static_cast<uint32_t>(gop)
                                   : static_cast<uint32_t>(std::max(1, static_cast<int>(std::lround(fps))));
    const uint32_t src_rate = static_cast<uint32_t>(std::max(1, static_cast<int>(std::lround(fps))));
    const uint32_t br       = static_cast<uint32_t>(std::max(1, bitrate_kbps));

    const uint64_t tid = static_cast<uint64_t>(pthread_self());
    rc                 = aclvencSetChannelDescThreadId(channel_desc_, tid);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescThreadId", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescCallback(channel_desc_, &AscendVencFfmpegMuxWriter::onVencCallback);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescCallback", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescEnType(channel_desc_, H264_MAIN_LEVEL);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescEnType", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescPicFormat(channel_desc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescPicFormat", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescPicWidth(channel_desc_, pic_w_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescPicWidth", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescPicHeight(channel_desc_, pic_h_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescPicHeight", rc);
        rollbackOpen();
        return false;
    }

    rc = aclvencSetChannelDescKeyFrameInterval(channel_desc_, gop_u);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescKeyFrameInterval", rc);
        rollbackOpen();
        return false;
    }

    rc = aclvencSetChannelDescRcMode(channel_desc_, 2u); // CBR
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescRcMode", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescSrcRate(channel_desc_, src_rate);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescSrcRate", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescMaxBitRate(channel_desc_, br);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescMaxBitRate", rc);
        rollbackOpen();
        return false;
    }

    rc = aclvencSetChannelDescBufAddr(channel_desc_, venc_out_buf_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescBufAddr", rc);
        rollbackOpen();
        return false;
    }
    rc = aclvencSetChannelDescBufSize(channel_desc_, venc_out_buf_size_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencSetChannelDescBufSize", rc);
        rollbackOpen();
        return false;
    }

    rc = aclvencCreateChannel(channel_desc_);
    if (rc != ACL_SUCCESS) {
        logVencOpenAclFailure("aclvencCreateChannel", rc);
        rollbackOpen();
        return false;
    }
    venc_channel_created_ = true;

    opened_.store(true, std::memory_order_release);
    LOG_INFO("AscendVencFfmpegMuxWriter: VENC+H264 mux opened device={} {}x{} fps={} gop={} bitrate_kbps={} url={}",
             device_id_, pic_w_, pic_h_, fps, gop_u, br, url);
    return true;
}

bool AscendVencFfmpegMuxWriter::write(const cv::Mat& frame) {
    if (!opened_.load(std::memory_order_acquire) || !channel_desc_ || frame.empty()) return false;

    ACL_CHECK_OR_RETURN(aclrtSetCurrentContext(ctx_), false);

    void* pic_dev = nullptr;
    aclError rc   = acldvppMalloc(&pic_dev, nv12_host_bytes_);
    if (rc != ACL_SUCCESS || !pic_dev) {
        LOG_ERROR("AscendVencFfmpegMuxWriter: acldvppMalloc pic failed ({})", static_cast<int>(rc));
        return false;
    }

    packSingleBgrToNv12Contiguous(frame, static_cast<int>(pic_w_), static_cast<int>(pic_h_), nv12_host_.data());

    rc = aclrtMemcpy(pic_dev, nv12_host_bytes_, nv12_host_.data(), nv12_host_bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("AscendVencFfmpegMuxWriter: H2D memcpy failed ({})", static_cast<int>(rc));
        (void)acldvppFree(pic_dev);
        return false;
    }

    acldvppPicDesc* pic_desc = acldvppCreatePicDesc();
    if (!pic_desc) {
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescData(pic_desc, pic_dev);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescSize(pic_desc, static_cast<uint32_t>(nv12_host_bytes_));
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescWidth(pic_desc, pic_w_);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescHeight(pic_desc, pic_h_);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescWidthStride(pic_desc, pic_w_);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    rc = acldvppSetPicDescHeightStride(pic_desc, pic_h_);
    if (rc != ACL_SUCCESS) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }

    aclvencFrameConfig* fc = aclvencCreateFrameConfig();
    if (!fc) {
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    (void)aclvencSetFrameConfigEos(fc, 0);
    (void)aclvencSetFrameConfigForceIFrame(fc, 0);

    frame_done_.store(false, std::memory_order_release);

    rc = aclvencSendFrame(channel_desc_, pic_desc, nullptr, fc, this);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("AscendVencFfmpegMuxWriter: aclvencSendFrame failed ({})", static_cast<int>(rc));
        (void)aclvencDestroyFrameConfig(fc);
        acldvppDestroyPicDesc(pic_desc);
        (void)acldvppFree(pic_dev);
        return false;
    }
    (void)aclvencDestroyFrameConfig(fc);

    // CANN6 VENC callback dispatch requires pumping report queue on the thread registered via
    // aclvencSetChannelDescThreadId(). SynchronizeStream/Device alone may not trigger callback delivery.
    (void)aclrtSynchronizeStream(stream_);
    for (int i = 0; i < 2000; ++i) {
        if (frame_done_.load(std::memory_order_acquire)) break;
        const aclError pr = aclrtProcessReport(1);
        if (pr != ACL_SUCCESS
#ifdef ACL_ERROR_RT_REPORT_TIMEOUT
            && pr != ACL_ERROR_RT_REPORT_TIMEOUT
#endif
        ) {
            LOG_WARN("AscendVencFfmpegMuxWriter: aclrtProcessReport failed ({})", static_cast<int>(pr));
        }
        (void)aclrtSynchronizeStream(stream_);
        (void)aclrtSynchronizeDevice();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!frame_done_.load(std::memory_order_acquire)) {
        LOG_WARN("AscendVencFfmpegMuxWriter: encode callback did not complete within timeout");
        return false;
    }

    return true;
}

void AscendVencFfmpegMuxWriter::close() {
    if (!opened_.load(std::memory_order_acquire) && !pipe_ && !channel_desc_ && !ctx_) {
        return;
    }
    rollbackOpen();
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
