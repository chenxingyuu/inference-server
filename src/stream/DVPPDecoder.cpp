#ifdef BUILD_ASCEND_BACKEND

#include "stream/DVPPDecoder.h"
#include "common/Logger.h"
#include "infer/AscendProcessRuntime.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <stdexcept>
#include <chrono>

// FFmpeg headers (demux only — no avcodec decode)
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace infer {

// Process-wide channel ID counter definition.
std::atomic<uint32_t> DVPPDecoder::s_channel_counter_{0};

// ── Lifecycle ──────────────────────────────────────────────────────────────

DVPPDecoder::DVPPDecoder()
    : channel_id_(s_channel_counter_.fetch_add(1, std::memory_order_relaxed))
{}

DVPPDecoder::~DVPPDecoder() {
    stop();
}

void DVPPDecoder::start(const StreamConfig& cfg, FrameCallback cb) {
    stream_id_  = cfg.id;
    device_id_  = cfg.ascend_device_id;
    stop_flag_.store(false, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(ctx_mu_);
        ctx_.device_id = device_id_;
        ctx_.cb        = std::move(cb);
    }

    startup_done_ = false;
    startup_ok_   = false;
    thread_ = std::thread(&DVPPDecoder::decodeLoop, this, cfg);

    // Wait until decodeLoop signals its startup result (success or permanent failure).
    // 15 s accommodates slow RTSP connections (stimeout=5s) plus DVPP channel init.
    std::unique_lock<std::mutex> lk(startup_mu_);
    startup_cv_.wait_for(lk, std::chrono::seconds(15),
                         [this] { return startup_done_; });
}

void DVPPDecoder::startForTest(FrameCallback cb) {
    stream_id_ = "test";
    stop_flag_.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(ctx_mu_);
        ctx_.cb        = std::move(cb);
        ctx_.device_id = 0;
    }

    thread_ = std::thread([this]() {
        running_.store(true, std::memory_order_release);
        while (!stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        running_.store(false, std::memory_order_release);
    });

    for (int i = 0; i < 1000 && !running_.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void DVPPDecoder::stop() {
    stop_flag_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
    running_.store(false, std::memory_order_relaxed);
}

// ── DVPP channel management ────────────────────────────────────────────────

bool DVPPDecoder::initChannel(int device_id, uint32_t aligned_w, uint32_t aligned_h,
                               bool is_h265) {
#if CANN_VERSION_MAJOR >= 7
    acldvppChannelDesc* desc = acldvppCreateChannelDesc();
    if (!desc) {
        LOG_ERROR("DVPPDecoder: acldvppCreateChannelDesc() returned null");
        return false;
    }
    channel_desc_ = desc;

    aclError err = aclrtCreateStream(&dvpp_stream_);
    if (err != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclrtCreateStream failed ({})", static_cast<int>(err));
        acldvppDestroyChannelDesc(channel_desc_);
        channel_desc_ = nullptr;
        return false;
    }

    aclError create_err;
    if (dvpp_api_.createChannel) {
        create_err = dvpp_api_.createChannel(&channel_desc_);
    } else {
        create_err = acldvppCreateChannel(channel_desc_);
    }
#else
    // CANN 6: aclvdec* API — codec type, output format and output dimensions
    // must all be declared on the channel descriptor before aclvdecCreateChannel().
    aclvdecChannelDesc* desc = aclvdecCreateChannelDesc();
    if (!desc) {
        LOG_ERROR("DVPPDecoder: aclvdecCreateChannelDesc() returned null");
        return false;
    }

    // Each DVPPDecoder instance uses a unique channel ID to prevent conflicts
    // when multiple streams run in parallel on the same device.
    aclvdecSetChannelDescChannelId(desc, channel_id_);

    // Codec type: must match the actual bitstream (H.264 or H.265).
    // CANN6 type is acldvppStreamFormat (H265_MAIN_LEVEL=0, H264_MAIN_LEVEL=2).
    const acldvppStreamFormat en_type = is_h265 ? H265_MAIN_LEVEL : H264_MAIN_LEVEL;
    aclvdecSetChannelDescEnType(desc, en_type);

    // Output pixel format: YUV420SP (NV12) — matches the static AIPP input_format.
    aclvdecSetChannelDescOutPicFormat(desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    // Output dimensions: DVPP uses these to size its internal decode buffers.
    aclvdecSetChannelDescOutPicWidth(desc, aligned_w);
    aclvdecSetChannelDescOutPicHeight(desc, aligned_h);

    aclvdecSetChannelDescCallback(desc, &DVPPDecoder::onDecoded);
    channel_desc_ = desc;

    aclError create_err;
    if (dvpp_api_.createChannel) {
        create_err = dvpp_api_.createChannel(&channel_desc_);
    } else {
        create_err = aclvdecCreateChannel(channel_desc_);
    }
#endif

    if (create_err != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: vdec channel create failed ({})",
                  static_cast<int>(create_err));
        destroyChannel();
        return false;
    }

    LOG_INFO("DVPPDecoder: DVPP vdec channel created, device={} channel_id={} "
             "codec={} aligned={}x{}",
             device_id, channel_id_, is_h265 ? "H265" : "H264", aligned_w, aligned_h);
    return true;
}

void DVPPDecoder::destroyChannel() {
    if (channel_desc_) {
        aclError err;
        if (dvpp_api_.destroyChannel) {
            err = dvpp_api_.destroyChannel(channel_desc_);
        } else {
#if CANN_VERSION_MAJOR >= 7
            err = acldvppDestroyChannel(channel_desc_);
#else
            err = aclvdecDestroyChannel(channel_desc_);
#endif
        }
        if (err != ACL_SUCCESS) {
            LOG_WARN("DVPPDecoder: vdec channel destroy failed ({})",
                     static_cast<int>(err));
        }
#if CANN_VERSION_MAJOR >= 7
        acldvppDestroyChannelDesc(channel_desc_);
#else
        aclvdecDestroyChannelDesc(channel_desc_);
#endif
        channel_desc_ = nullptr;
    }
    if (dvpp_stream_) {
        aclrtDestroyStream(dvpp_stream_);
        dvpp_stream_ = nullptr;
    }
}

// ── DVPP decode callback ────────────────────────────────────────────────────
//
// CANN6: aclvdecSendFrame is ASYNCHRONOUS. The callback fires on a DVPP
// internal thread when the hardware decode completes. Lifetime rules:
//   - stream_desc (input): owned by DVPP until this callback; destroyed HERE.
//   - bitstream_dev: device memory for compressed data; freed HERE via FrameCtx.
//   - output YUV buffer (yuv_buf): owned by frame_ref; freed when Frame is dropped.
//   - pic_desc (output): destroyed in the frame_ref deleter.
//
// user_data = heap-allocated FrameCtx* (deleted here after callback returns).

void DVPPDecoder::onDecoded(acldvppStreamDesc* input,
                             acldvppPicDesc*    output,
                             void*              user_data) {
    if (!user_data) return;
    auto* fctx = static_cast<FrameCtx*>(user_data);

    // Destroy the stream descriptor — DVPP passes it back here after decode completes.
    // This MUST happen here, not immediately after aclvdecSendFrame (the call is async).
    if (input) {
        acldvppDestroyStreamDesc(input);
    }

    // Release the compressed bitstream device memory.
    if (fctx->bitstream_dev) {
        acldvppFree(fctx->bitstream_dev);
        fctx->bitstream_dev = nullptr;
    }

    AscendBuffer buf;
    buf.device_id     = fctx->device_id;
    buf.aligned_width  = static_cast<int>(fctx->aligned_width);
    buf.aligned_height = static_cast<int>(fctx->aligned_height);

    if (output) {
        buf.yuv_device = acldvppGetPicDescData(output);
        buf.width      = fctx->codec_width;
        buf.height     = fctx->codec_height;

        // frame_ref deleter returns the YUV buffer and destroys the pic desc.
        acldvppPicDesc* desc_copy = output;
        buf.frame_ref = std::shared_ptr<void>(
            reinterpret_cast<void*>(1),
            [desc_copy](void*) {
                void* data = acldvppGetPicDescData(desc_copy);
                if (data) {
                    aclError rc = acldvppFree(data);
                    if (rc != ACL_SUCCESS) {
                        LOG_WARN("DVPPDecoder: acldvppFree in callback failed ({})",
                                 static_cast<int>(rc));
                    }
                }
                aclError rc = acldvppDestroyPicDesc(desc_copy);
                if (rc != ACL_SUCCESS) {
                    LOG_WARN("DVPPDecoder: acldvppDestroyPicDesc in callback failed ({})",
                             static_cast<int>(rc));
                }
            }
        );
    }

    const double decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - fctx->submit_ts).count();
    LOG_DEBUG("DVPPDecoder: hw decode done {}x{} decode_ms={:.1f}",
              buf.width, buf.height, decode_ms);

    Frame f;
    f.ascend_buf          = std::move(buf);
    f.is_ascend           = true;
    f.meta.stream_id      = fctx->stream_id;
    f.meta.frame_seq      = fctx->frame_seq;
    f.meta.capture_ts     = fctx->capture_ts;
    f.meta.capture_mono_ns = fctx->capture_mono_ns;
    f.meta.orig_width     = fctx->codec_width;
    f.meta.orig_height    = fctx->codec_height;

    if (fctx->cb) fctx->cb(std::move(f));

    delete fctx;
}

// ── Decode loop ────────────────────────────────────────────────────────────

void DVPPDecoder::decodeLoop(StreamConfig cfg) {
    // Ensure aclInit() has been called before any ACL API usage.
    AscendProcessRuntime::acquire();
    struct RuntimeGuard {
        ~RuntimeGuard() { AscendProcessRuntime::release(); }
    } runtime_guard;

    // On any early return, signal startup failure so start() doesn't wait forever.
    // Cancelled (set to true) once we successfully enter the packet decode loop.
    struct StartupFailGuard {
        DVPPDecoder& dec;
        bool cancelled{false};
        ~StartupFailGuard() {
            if (cancelled) return;
            std::lock_guard<std::mutex> lk(dec.startup_mu_);
            dec.startup_done_ = true;
            dec.startup_ok_   = false;
            dec.startup_cv_.notify_one();
        }
    } startup_guard{*this};

    aclError rc = aclrtSetDevice(device_id_);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclrtSetDevice({}) failed: {}", device_id_,
                  static_cast<int>(rc));
        return;
    }
    aclrtContext ctx_handle = nullptr;
    rc = aclrtCreateContext(&ctx_handle, device_id_);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclrtCreateContext failed: {}", static_cast<int>(rc));
        return;
    }

    const int reconnect_delay_ms =
        cfg.reconnect_delay_ms > 0 ? cfg.reconnect_delay_ms : 2000;
    bool first_connect = true;

    // ── Outer reconnect loop ──────────────────────────────────────────────────
    // Each iteration: open RTSP → init DVPP channel → decode packets.
    // On channel poisoning (consecutive sendFrame failures), loop back and reconnect.
    while (!stop_flag_.load(std::memory_order_acquire)) {

    // ── Open RTSP and detect stream geometry BEFORE creating DVPP channel ────
    // initChannel requires codec type and aligned resolution, which are only
    // known after avformat_find_stream_info().
    AVFormatContext* fmt_ctx = nullptr;
    AVDictionary*    opts    = nullptr;
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);

    if (avformat_open_input(&fmt_ctx, cfg.url.c_str(), nullptr, &opts) < 0) {
        av_dict_free(&opts);
        if (first_connect) {
            LOG_ERROR("DVPPDecoder: cannot open RTSP: {}", cfg.url);
            aclrtDestroyContext(ctx_handle);
            return;
        }
        LOG_WARN("DVPPDecoder [{}]: cannot open RTSP, retrying in {}ms",
                 stream_id_, reconnect_delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }
    av_dict_free(&opts);
    avformat_find_stream_info(fmt_ctx, nullptr);

    int video_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = static_cast<int>(i);
            break;
        }
    }
    if (video_idx < 0) {
        LOG_ERROR("DVPPDecoder [{}]: no video stream in {}", stream_id_, cfg.url);
        avformat_close_input(&fmt_ctx);
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }

    // Read stream geometry and compute DVPP-aligned dimensions.
    codec_width_    = static_cast<uint32_t>(fmt_ctx->streams[video_idx]->codecpar->width);
    codec_height_   = static_cast<uint32_t>(fmt_ctx->streams[video_idx]->codecpar->height);
    aligned_width_  = (codec_width_  + 15u) & ~15u;
    aligned_height_ = (codec_height_ +  1u) & ~1u;

    // Ascend 310P VDEC hardware limits: [128, 4096] for both dimensions.
    // Check after alignment so the error message reflects what DVPP actually sees.
    static constexpr uint32_t kDvppMinDim = 128u;
    static constexpr uint32_t kDvppMaxDim = 4096u;
    if (aligned_width_  < kDvppMinDim || aligned_width_  > kDvppMaxDim ||
        aligned_height_ < kDvppMinDim || aligned_height_ > kDvppMaxDim) {
        LOG_ERROR("DVPPDecoder [{}]: resolution {}x{} (aligned {}x{}) out of DVPP range "
                  "[{}–{}] × [{}–{}] — stream cannot use hardware decode",
                  stream_id_, codec_width_, codec_height_,
                  aligned_width_, aligned_height_,
                  kDvppMinDim, kDvppMaxDim, kDvppMinDim, kDvppMaxDim);
        avformat_close_input(&fmt_ctx);
        aclrtDestroyContext(ctx_handle);
        return;
    }

    const AVCodecID codec_id = fmt_ctx->streams[video_idx]->codecpar->codec_id;
    const bool is_h265 = (codec_id == AV_CODEC_ID_HEVC);
    if (codec_id != AV_CODEC_ID_H264 && codec_id != AV_CODEC_ID_HEVC) {
        LOG_ERROR("DVPPDecoder [{}]: unsupported codec {} (only H.264/H.265 supported)",
                  stream_id_, static_cast<int>(codec_id));
        avformat_close_input(&fmt_ctx);
        aclrtDestroyContext(ctx_handle);
        return;
    }

    LOG_INFO("DVPPDecoder [{}]: stream {}x{} (aligned {}x{}) codec={}",
             stream_id_, codec_width_, codec_height_, aligned_width_, aligned_height_,
             is_h265 ? "H265" : "H264");

    if (!initChannel(device_id_, aligned_width_, aligned_height_, is_h265)) {
        avformat_close_input(&fmt_ctx);
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }

    // ── Annex B bitstream filter ──────────────────────────────────────────
    // DVPP requires H.264/H.265 in Annex B format (start codes).
    // Sources served via RTSP from an MP4 file (or certain IP cameras) may
    // deliver packets in AVCC format (length-prefix) with SPS/PPS only in
    // AVCodecParameters::extradata.  h264_mp4toannexb / hevc_mp4toannexb:
    //   - converts length prefixes to start codes, AND
    //   - prepends SPS/PPS from extradata to every key frame.
    // The filter is a no-op when the stream is already in Annex B.
    const char* bsf_name = is_h265 ? "hevc_mp4toannexb" : "h264_mp4toannexb";
    const AVBitStreamFilter* bsf_filter = av_bsf_get_by_name(bsf_name);
    AVBSFContext* bsf_ctx = nullptr;
    if (bsf_filter) {
        if (av_bsf_alloc(bsf_filter, &bsf_ctx) == 0) {
            avcodec_parameters_copy(bsf_ctx->par_in,
                                    fmt_ctx->streams[video_idx]->codecpar);
            bsf_ctx->time_base_in = fmt_ctx->streams[video_idx]->time_base;
            if (av_bsf_init(bsf_ctx) < 0) {
                LOG_WARN("DVPPDecoder: {} init failed, sending raw packets to DVPP",
                         bsf_name);
                av_bsf_free(&bsf_ctx);
                bsf_ctx = nullptr;
            } else {
                LOG_INFO("DVPPDecoder: {} BSF active (AVCC→Annex B conversion)",
                         bsf_name);
            }
        }
    }

    if (first_connect) {
        running_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(startup_mu_);
            startup_done_ = true;
            startup_ok_   = true;
            startup_cv_.notify_one();
        }
        startup_guard.cancelled = true;
        first_connect = false;
    }
    LOG_INFO("DVPPDecoder [{}]: starting packet decode loop", stream_id_);

    // ── Single-packet DVPP submit helper ─────────────────────────────────
    //
    // CANN6 aclvdecSendFrame is ASYNCHRONOUS: the call returns as soon as the
    // frame is queued; the decode callback fires on a DVPP worker thread when
    // the hardware finishes. Therefore:
    //   - bitstream_dev lifetime: managed by FrameCtx, freed inside onDecoded
    //   - yuv_buf lifetime: managed by frame_ref deleter in onDecoded
    //   - NO aclrtSynchronizeDevice() needed here (it would not wait for VDEC callbacks)
    auto submitToDvpp = [&](AVPacket* raw_pkt) -> bool {
        void* bitstream_dev = nullptr;
        aclError bitstream_rc = acldvppMalloc(&bitstream_dev,
                                               static_cast<size_t>(raw_pkt->size));
        if (bitstream_rc != ACL_SUCCESS || !bitstream_dev) {
            LOG_ERROR("DVPPDecoder: bitstream acldvppMalloc failed (rc={}, size={})",
                      static_cast<int>(bitstream_rc), raw_pkt->size);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }

        aclError copy_rc = aclrtMemcpy(bitstream_dev,
                                        static_cast<size_t>(raw_pkt->size),
                                        raw_pkt->data,
                                        static_cast<size_t>(raw_pkt->size),
                                        ACL_MEMCPY_HOST_TO_DEVICE);
        if (copy_rc != ACL_SUCCESS) {
            LOG_ERROR("DVPPDecoder: bitstream H2D copy failed (rc={}, size={})",
                      static_cast<int>(copy_rc), raw_pkt->size);
            acldvppFree(bitstream_dev);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }

        acldvppStreamDesc* stream_desc = acldvppCreateStreamDesc();
        if (!stream_desc) {
            LOG_ERROR("DVPPDecoder: acldvppCreateStreamDesc returned null");
            acldvppFree(bitstream_dev);
            return false;
        }
        acldvppSetStreamDescData(stream_desc, bitstream_dev);
        acldvppSetStreamDescSize(stream_desc, static_cast<uint32_t>(raw_pkt->size));
        // EOS=0: explicitly mark each frame as not end-of-stream.
        // Some CANN6 driver builds treat the default (unset) as EOS, causing stalls.
        acldvppSetStreamDescEos(stream_desc, 0);

        acldvppPicDesc* pic_desc = acldvppCreatePicDesc();
        if (!pic_desc) {
            LOG_ERROR("DVPPDecoder: acldvppCreatePicDesc returned null");
            acldvppFree(bitstream_dev);
            acldvppDestroyStreamDesc(stream_desc);
            return false;
        }

        const uint32_t yuv_size = alignedYuvSize(codec_width_, codec_height_);
        void* yuv_buf = nullptr;
        aclError malloc_rc = acldvppMalloc(&yuv_buf, yuv_size);
        if (malloc_rc != ACL_SUCCESS || !yuv_buf) {
            LOG_ERROR("DVPPDecoder: acldvppMalloc for YUV failed (rc={}, size={})",
                      static_cast<int>(malloc_rc), yuv_size);
            acldvppDestroyPicDesc(pic_desc);
            acldvppFree(bitstream_dev);
            acldvppDestroyStreamDesc(stream_desc);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }

        acldvppSetPicDescData(pic_desc, yuv_buf);
        acldvppSetPicDescSize(pic_desc, yuv_size);
        acldvppSetPicDescWidth(pic_desc, codec_width_);
        acldvppSetPicDescHeight(pic_desc, codec_height_);
        acldvppSetPicDescWidthStride(pic_desc, aligned_width_);
        acldvppSetPicDescHeightStride(pic_desc, aligned_height_);
        acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

        FrameCtx* fctx;
        {
            std::lock_guard<std::mutex> lk(ctx_mu_);
            const auto now_steady = std::chrono::steady_clock::now();
            fctx = new FrameCtx{};
            fctx->cb              = ctx_.cb;
            fctx->device_id       = ctx_.device_id;
            fctx->bitstream_dev   = bitstream_dev;
            fctx->submit_ts       = now_steady;
            fctx->stream_id       = stream_id_;
            fctx->frame_seq       = frame_seq_++;
            fctx->aligned_width   = aligned_width_;
            fctx->aligned_height  = aligned_height_;
            fctx->codec_width     = static_cast<int>(codec_width_);
            fctx->codec_height    = static_cast<int>(codec_height_);
            fctx->capture_ts      = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            fctx->capture_mono_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now_steady.time_since_epoch()).count());
        }

        aclError send_rc = ACL_ERROR_FAILURE;
#if CANN_VERSION_MAJOR >= 7
        auto vdec = dvpp_api_.vdecProcess ? dvpp_api_.vdecProcess
            : VdecProcessFn([](AclVdecChannelDesc* ch, acldvppStreamDesc* sd,
                 acldvppPicDesc* pd, aclrtStream s,
                 AclVdecCb fn, void* ud) -> aclError {
                return acldvppVdecProcess(ch, sd, pd, s, fn, ud);
            });
        send_rc = vdec(channel_desc_, stream_desc, pic_desc, dvpp_stream_,
                       &DVPPDecoder::onDecoded, fctx);
#else
        auto vdec = dvpp_api_.vdecProcess ? dvpp_api_.vdecProcess
            : VdecProcessFn([](AclVdecChannelDesc* ch, acldvppStreamDesc* sd,
                 acldvppPicDesc* pd, aclvdecFrameConfig* cfg, void* ud) -> aclError {
                return aclvdecSendFrame(ch, sd, pd, cfg, ud);
            });
        aclvdecFrameConfig* frame_cfg = aclvdecCreateFrameConfig();
        if (!frame_cfg) {
            LOG_ERROR("DVPPDecoder: aclvdecCreateFrameConfig returned null");
            delete fctx;
            acldvppFree(bitstream_dev);
            acldvppFree(yuv_buf);
            acldvppDestroyPicDesc(pic_desc);
            acldvppDestroyStreamDesc(stream_desc);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }
        send_rc = vdec(channel_desc_, stream_desc, pic_desc, frame_cfg, fctx);
        aclvdecDestroyFrameConfig(frame_cfg);
#endif
        if (send_rc != ACL_SUCCESS) {
            LOG_ERROR("DVPPDecoder: sendFrame failed (rc={})",
                      static_cast<int>(send_rc));
            // On failure DVPP will NOT invoke the callback, so we own all resources.
            acldvppDestroyStreamDesc(stream_desc);
            acldvppFree(bitstream_dev);
            delete fctx;
            acldvppFree(yuv_buf);
            acldvppDestroyPicDesc(pic_desc);
            return false;
        }
        // On success: stream_desc is owned by DVPP until the callback fires.
        //   onDecoded() destroys stream_desc via its `input` parameter.
        //   bitstream_dev and yuv_buf/pic_desc are owned by FrameCtx / onDecoded.
        return true;
    };

    // ── Packet decode loop ────────────────────────────────────────────────
    // On 5 consecutive sendFrame failures the channel is poisoned; break out
    // so the outer reconnect loop can tear down and recreate everything.
    AVPacket* pkt = av_packet_alloc();
    int consecutive_failures = 0;
    constexpr int kMaxConsecutiveFailures = 5;
    bool channel_poisoned = false;

    while (!stop_flag_.load(std::memory_order_acquire)) {
        int ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) break;
            LOG_WARN("DVPPDecoder: av_read_frame error: {}", ret);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (pkt->stream_index != video_idx) {
            av_packet_unref(pkt);
            continue;
        }

        bool send_ok = false;
        if (bsf_ctx) {
            // av_bsf_send_packet transfers packet ownership to the BSF context.
            // After this call pkt is in a blank/reset state — do NOT unref.
            if (av_bsf_send_packet(bsf_ctx, pkt) < 0) {
                av_packet_unref(pkt);  // safety unref on error path
                continue;
            }
            // One input packet may yield one or more Annex-B output packets.
            while (av_bsf_receive_packet(bsf_ctx, pkt) == 0) {
                if (submitToDvpp(pkt)) send_ok = true;
                av_packet_unref(pkt);
            }
        } else {
            send_ok = submitToDvpp(pkt);
            av_packet_unref(pkt);
        }

        if (!send_ok) {
            if (++consecutive_failures >= kMaxConsecutiveFailures) {
                LOG_ERROR("DVPPDecoder [{}]: {} consecutive sendFrame failures — "
                          "channel poisoned (aicpu exception), reconnecting",
                          stream_id_, consecutive_failures);
                channel_poisoned = true;
                break;
            }
        } else {
            consecutive_failures = 0;
        }
    }

    // ── Per-connection cleanup ────────────────────────────────────────────
    av_packet_free(&pkt);
    if (bsf_ctx) av_bsf_free(&bsf_ctx);
    avformat_close_input(&fmt_ctx);
    // aclvdecDestroyChannel blocks until all in-flight callbacks complete (CANN6 guarantee).
    destroyChannel();

    if (!channel_poisoned || stop_flag_.load(std::memory_order_acquire)) {
        break;  // EOF or graceful stop — exit reconnect loop
    }

    LOG_WARN("DVPPDecoder [{}]: sleeping {}ms before reconnect",
             stream_id_, reconnect_delay_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
    // Continue outer reconnect loop

    } // end while (!stop_flag_) reconnect loop

    running_.store(false, std::memory_order_release);
    aclrtDestroyContext(ctx_handle);
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
