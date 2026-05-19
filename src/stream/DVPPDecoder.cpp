#ifdef BUILD_ASCEND_BACKEND

#include "stream/DVPPDecoder.h"
#include "common/Logger.h"
#include "infer/AscendProcessRuntime.h"
#include "stream/DvppOutputPool.h"

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
                               int h264_profile, bool is_h265) {
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

    // Codec type: must exactly match the actual bitstream profile.
    // CANN6 silently drops frames (no callback) when the profile doesn't match.
    // acldvppStreamFormat: H265_MAIN_LEVEL=0, H264_BASELINE_LEVEL=1,
    //                       H264_MAIN_LEVEL=2, H264_HIGH_LEVEL=3.
    acldvppStreamFormat en_type;
    if (is_h265) {
        en_type = H265_MAIN_LEVEL;
    } else if (h264_profile > 0 && h264_profile < 77) {
        en_type = H264_BASELINE_LEVEL;   // profile_idc=66 (Baseline/Constrained Baseline)
    } else if (h264_profile > 0 && h264_profile < 100) {
        en_type = H264_MAIN_LEVEL;       // profile_idc=77/88 (Main/Extended)
    } else {
        en_type = H264_HIGH_LEVEL;       // profile_idc=100+ (High) or unknown
    }
    LOG_INFO("DVPPDecoder: DVPP profile selected: {} (h264_profile={})",
             static_cast<int>(en_type), h264_profile);
    aclvdecSetChannelDescEnType(desc, en_type);

    // Output pixel format: YUV420SP (NV12) — matches the static AIPP input_format.
    aclvdecSetChannelDescOutPicFormat(desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    // Output dimensions: DVPP uses these to size its internal decode buffers.
    aclvdecSetChannelDescOutPicWidth(desc, aligned_w);
    aclvdecSetChannelDescOutPicHeight(desc, aligned_h);

    // Bind the calling thread (decodeLoop) as the callback-dispatch thread.
    // CANN 6 vdec hardware completion is delivered through this thread's aclrt
    // report queue; without this binding the decode callback is NEVER invoked,
    // even though aclvdecSendFrame succeeds. Must be set BEFORE aclvdecCreateChannel.
    // Mirrors the VENC pattern used in AscendVencFfmpegMuxWriter::open().
    const uint64_t cb_tid = static_cast<uint64_t>(pthread_self());
    aclError tid_rc = aclvdecSetChannelDescThreadId(desc, cb_tid);
    if (tid_rc != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclvdecSetChannelDescThreadId failed ({})",
                  static_cast<int>(tid_rc));
        aclvdecDestroyChannelDesc(desc);
        return false;
    }

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
//   - output pic_desc + yuv_buf: owned by the DvppOutputPool slot; NOT freed here.
//     frame_ref deleter calls pool_release (returns slot to pool and resets desc).
//
// user_data = heap-allocated FrameCtx* (deleted here after callback returns).

void DVPPDecoder::onDecoded(acldvppStreamDesc* input,
                             acldvppPicDesc*    output,
                             void*              user_data) {
    if (!user_data) return;
    auto* fctx = static_cast<FrameCtx*>(user_data);
    if (fctx->acl_context) {
        (void)aclrtSetCurrentContext(fctx->acl_context);
    }
    LOG_DEBUG("DVPPDecoder [{}]: onDecoded fired seq={} output={}",
              fctx->stream_id, fctx->frame_seq, output ? "ok" : "null");

    // Destroy the stream descriptor — DVPP passes it back here after decode completes.
    if (input) acldvppDestroyStreamDesc(input);

    // Release the compressed bitstream device memory.
    if (fctx->bitstream_dev) {
        acldvppFree(fctx->bitstream_dev);
        fctx->bitstream_dev = nullptr;
    }

    AscendBuffer buf;
    buf.device_id = fctx->device_id;

    const auto t_cb = std::chrono::steady_clock::now();
    const double decode_ms =
        std::chrono::duration<double, std::milli>(t_cb - fctx->submit_ts).count();

    if (output && fctx->slot_yuv_buf && fctx->in_pic_desc) {
        if (fctx->vpc_enabled && fctx->vpc_pool && fctx->vpc_scaler) {
            DvppOutputPool::Slot* vpc_slot = fctx->vpc_pool->acquire(200);
            if (!vpc_slot) {
                LOG_WARN("DVPPDecoder [{}]: VPC output pool exhausted, dropping frame",
                         fctx->stream_id);
                fctx->pool_release.reset();
                delete fctx;
                return;
            }
            fctx->vpc_pool->resetDesc(*vpc_slot);

            const auto t_vpc0 = std::chrono::steady_clock::now();
            const bool vpc_ok =
                fctx->vpc_scaler->resize(fctx->in_pic_desc, vpc_slot->pic_desc);
            const double vpc_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_vpc0).count();
            fctx->pool_release.reset();

            if (!vpc_ok) {
                fctx->vpc_pool->release(vpc_slot);
                delete fctx;
                return;
            }

            const uint32_t out_aw = (fctx->vpc_out_w + 15u) & ~15u;
            const uint32_t out_ah = (fctx->vpc_out_h +  1u) & ~1u;
            buf.yuv_device      = vpc_slot->yuv_buf;
            buf.width           = static_cast<int>(fctx->vpc_out_w);
            buf.height          = static_cast<int>(fctx->vpc_out_h);
            buf.aligned_width   = static_cast<int>(out_aw);
            buf.aligned_height  = static_cast<int>(out_ah);

            auto pool_sp = fctx->vpc_pool;
            buf.frame_ref = std::shared_ptr<void>(
                reinterpret_cast<void*>(1),
                [pool_sp, vpc_slot](void*) { pool_sp->release(vpc_slot); });

            LOG_DEBUG("DVPPDecoder [{}]: vpc resize {}x{} -> {}x{} decode_ms={:.1f} vpc_ms={:.1f}",
                      fctx->stream_id, fctx->codec_width, fctx->codec_height,
                      fctx->vpc_out_w, fctx->vpc_out_h, decode_ms, vpc_ms);
        } else {
            buf.yuv_device     = fctx->slot_yuv_buf;
            buf.width          = fctx->codec_width;
            buf.height         = fctx->codec_height;
            buf.aligned_width  = static_cast<int>(fctx->aligned_width);
            buf.aligned_height = static_cast<int>(fctx->aligned_height);
            buf.frame_ref      = std::move(fctx->pool_release);
            LOG_DEBUG("DVPPDecoder: hw decode done {}x{} decode_ms={:.1f}",
                      buf.width, buf.height, decode_ms);
        }
    }

    Frame f;
    f.ascend_buf           = std::move(buf);
    f.is_ascend            = true;
    f.meta.stream_id       = fctx->stream_id;
    f.meta.frame_seq       = fctx->frame_seq;
    f.meta.capture_ts      = fctx->capture_ts;
    f.meta.capture_mono_ns = fctx->capture_mono_ns;
    f.meta.orig_width      = fctx->codec_width;
    f.meta.orig_height     = fctx->codec_height;

    if (fctx->cb) fctx->cb(std::move(f));

    delete fctx;
}

// ── Decode loop ────────────────────────────────────────────────────────────

void DVPPDecoder::decodeLoop(StreamConfig cfg) {
    vpc_out_w_ = cfg.ascend_vpc_out_width > 0
        ? static_cast<uint32_t>(cfg.ascend_vpc_out_width) : 0;
    vpc_out_h_ = cfg.ascend_vpc_out_height > 0
        ? static_cast<uint32_t>(cfg.ascend_vpc_out_height) : 0;

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
    rc = aclrtSetCurrentContext(ctx_handle);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclrtSetCurrentContext failed: {}", static_cast<int>(rc));
        aclrtDestroyContext(ctx_handle);
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

    const int h264_profile = is_h265 ? -1
        : fmt_ctx->streams[video_idx]->codecpar->profile;
    LOG_INFO("DVPPDecoder [{}]: stream {}x{} (aligned {}x{}) codec={} profile={}",
             stream_id_, codec_width_, codec_height_, aligned_width_, aligned_height_,
             is_h265 ? "H265" : "H264", h264_profile);

    if (aclrtSetCurrentContext(ctx_handle) != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder [{}]: aclrtSetCurrentContext failed before channel init",
                  stream_id_);
        avformat_close_input(&fmt_ctx);
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }

    if (!initChannel(device_id_, aligned_width_, aligned_height_, h264_profile, is_h265)) {
        avformat_close_input(&fmt_ctx);
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }

    // ── Annex B bitstream filter ──────────────────────────────────────────
    // DVPP requires H.264/H.265 in Annex B format (start codes).
    //
    // mp4toannexb is needed only for file-based sources (MP4/MKV) where both
    // extradata AND packet payloads are in AVCC (length-prefix) format.
    //
    // For RTSP/RTP: av_read_frame returns packets already in Annex B (start
    // codes), but the SDP-derived extradata has AVCC layout (extradata[0]==1).
    // The BSF sees AVCC extradata, sets length_size=4, then misinterprets the
    // 0x00000001 start code as length=1 — producing 1-byte mangled NAL units.
    // DVPP silently drops the corrupted bitstream and never fires the callback.
    const char* fmt_name = fmt_ctx->iformat ? fmt_ctx->iformat->name : "";
    const bool is_rtp_based = (strstr(fmt_name, "rtsp") != nullptr ||
                                strstr(fmt_name, "rtp")  != nullptr);
    AVBSFContext* bsf_ctx = nullptr;
    if (!is_rtp_based) {
        const char* bsf_name = is_h265 ? "hevc_mp4toannexb" : "h264_mp4toannexb";
        const AVBitStreamFilter* bsf_filter = av_bsf_get_by_name(bsf_name);
        if (bsf_filter && av_bsf_alloc(bsf_filter, &bsf_ctx) == 0) {
            avcodec_parameters_copy(bsf_ctx->par_in,
                                    fmt_ctx->streams[video_idx]->codecpar);
            bsf_ctx->time_base_in = fmt_ctx->streams[video_idx]->time_base;
            if (av_bsf_init(bsf_ctx) < 0) {
                LOG_WARN("DVPPDecoder: {} init failed, sending raw packets to DVPP",
                         bsf_name);
                av_bsf_free(&bsf_ctx);
                bsf_ctx = nullptr;
            } else {
                LOG_INFO("DVPPDecoder: {} BSF active (file source, AVCC→Annex B)",
                         bsf_name);
            }
        }
    } else {
        LOG_INFO("DVPPDecoder [{}]: RTSP/RTP input ({}) — packets already Annex B, "
                 "BSF skipped", stream_id_, fmt_name);
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
    LOG_INFO("DVPPDecoder [{}]: starting packet decode loop (CANN_VERSION_MAJOR={})",
             stream_id_, CANN_VERSION_MAJOR);

    // ── CANN 6: callback dispatch ─────────────────────────────────────────
    // No aclrtSubscribeReport here — vdec callbacks are routed via the thread
    // registered by aclvdecSetChannelDescThreadId() in initChannel (this same
    // decodeLoop thread). We just need to pump aclrtProcessReport() below.

    // ── Pre-allocate output buffer pool ──────────────────────────────────
    // CANN 6 requires output acldvppPicDesc params to be explicitly reset before
    // each aclvdecSendFrame. Reusing pool slots (rather than malloc/free per frame)
    // ensures params are always reset and bounds DVPP memory usage.
    const uint32_t yuv_size = alignedYuvSize(codec_width_, codec_height_);
    auto output_pool = std::make_shared<DvppOutputPool>();
    if (!output_pool->init(yuv_size, codec_width_, codec_height_,
                           aligned_width_, aligned_height_)) {
        LOG_ERROR("DVPPDecoder [{}]: failed to allocate output buffer pool", stream_id_);
        if (bsf_ctx) av_bsf_free(&bsf_ctx);
        avformat_close_input(&fmt_ctx);
        destroyChannel();
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }
    LOG_INFO("DVPPDecoder [{}]: output pool allocated ({} slots, {} bytes each)",
             stream_id_, DvppOutputPool::kSize, yuv_size);

    vpc_enabled_ = vpc_out_w_ > 0 && vpc_out_h_ > 0 &&
        (vpc_out_w_ != codec_width_ || vpc_out_h_ != codec_height_);

    std::shared_ptr<DvppOutputPool> vpc_output_pool;
    DvppVpcScaler* vpc_scaler_ptr =
        vpc_scaler_override_ ? vpc_scaler_override_ : &vpc_scaler_;

    if (vpc_enabled_) {
        if (!vpc_scaler_ptr->init(device_id_, ctx_handle)) {
            LOG_ERROR("DVPPDecoder [{}]: DvppVpcScaler init failed", stream_id_);
            output_pool->destroy();
            if (bsf_ctx) av_bsf_free(&bsf_ctx);
            avformat_close_input(&fmt_ctx);
            destroyChannel();
            if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
            continue;
        }
        const uint32_t vpc_aw = (vpc_out_w_ + 15u) & ~15u;
        const uint32_t vpc_ah = (vpc_out_h_ +  1u) & ~1u;
        const uint32_t vpc_yuv = alignedYuvSize(vpc_out_w_, vpc_out_h_);
        vpc_output_pool = std::make_shared<DvppOutputPool>();
        if (!vpc_output_pool->init(vpc_yuv, vpc_out_w_, vpc_out_h_, vpc_aw, vpc_ah)) {
            LOG_ERROR("DVPPDecoder [{}]: failed to allocate VPC output pool", stream_id_);
            vpc_scaler_ptr->shutdown();
            output_pool->destroy();
            if (bsf_ctx) av_bsf_free(&bsf_ctx);
            avformat_close_input(&fmt_ctx);
            destroyChannel();
            if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
            continue;
        }
        LOG_INFO("DVPPDecoder [{}]: VPC enabled {}x{} -> {}x{} ({} bytes/slot)",
                 stream_id_, codec_width_, codec_height_, vpc_out_w_, vpc_out_h_, vpc_yuv);
    } else {
        vpc_scaler_.shutdown();
    }

    // ── CANN 6: one aclvdecFrameConfig shared across all frames in this session ─
    // CANN 6 aclvdecSendFrame stores the frame_cfg POINTER internally until the
    // decode callback fires. Destroying frame_cfg immediately after each send
    // creates a dangling pointer, causing DVPP to access freed memory — which
    // silently prevents the callback from ever being invoked.
    // Fix: allocate once per channel session, destroy after destroyChannel() drains.
#if CANN_VERSION_MAJOR < 7
    aclvdecFrameConfig* session_frame_cfg = aclvdecCreateFrameConfig();
    if (!session_frame_cfg) {
        LOG_ERROR("DVPPDecoder [{}]: aclvdecCreateFrameConfig failed", stream_id_);
        if (bsf_ctx) av_bsf_free(&bsf_ctx);
        avformat_close_input(&fmt_ctx);
        destroyChannel();
        if (first_connect) { aclrtDestroyContext(ctx_handle); return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms));
        continue;
    }
#endif

    // ── Single-packet DVPP submit helper ─────────────────────────────────
    //
    // CANN6 aclvdecSendFrame is ASYNCHRONOUS: the call returns as soon as the
    // frame is queued; the decode callback fires on a DVPP worker thread when
    // the hardware finishes. Lifetime rules:
    //   - bitstream_dev: freed inside onDecoded (per-frame allocation)
    //   - yuv_buf / pic_desc: owned by the output pool; returned via frame_ref deleter
    //   - session_frame_cfg: shared across all frames; destroyed after draining channel
    auto submitToDvpp = [&](AVPacket* raw_pkt) -> bool {
        // Acquire a free output slot (blocks up to 200 ms if pool is full).
        DvppOutputPool::Slot* slot = output_pool->acquire(200);
        if (!slot) {
            LOG_WARN("DVPPDecoder [{}]: output pool exhausted, dropping frame", stream_id_);
            return true;  // not a channel error — treat as soft drop
        }
        // Always reset desc params before use (mandatory per CANN 6 docs).
        output_pool->resetDesc(*slot);

        // Log first 3 packets to verify DVPP receives valid Annex B start codes.
        if (frame_seq_ < 3 && raw_pkt->size >= 8) {
            LOG_INFO("DVPPDecoder [{}]: pkt#{} size={} first8={:02x} {:02x} {:02x} {:02x}"
                     " {:02x} {:02x} {:02x} {:02x}",
                     stream_id_, frame_seq_, raw_pkt->size,
                     raw_pkt->data[0], raw_pkt->data[1],
                     raw_pkt->data[2], raw_pkt->data[3],
                     raw_pkt->data[4], raw_pkt->data[5],
                     raw_pkt->data[6], raw_pkt->data[7]);
        }

        void* bitstream_dev = nullptr;
        aclError bitstream_rc = acldvppMalloc(&bitstream_dev,
                                               static_cast<size_t>(raw_pkt->size));
        if (bitstream_rc != ACL_SUCCESS || !bitstream_dev) {
            LOG_ERROR("DVPPDecoder: bitstream acldvppMalloc failed (rc={}, size={})",
                      static_cast<int>(bitstream_rc), raw_pkt->size);
            output_pool->release(slot);
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
            output_pool->release(slot);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }

        acldvppStreamDesc* stream_desc = acldvppCreateStreamDesc();
        if (!stream_desc) {
            LOG_ERROR("DVPPDecoder: acldvppCreateStreamDesc returned null");
            acldvppFree(bitstream_dev);
            output_pool->release(slot);
            return false;
        }
        acldvppSetStreamDescData(stream_desc, bitstream_dev);
        acldvppSetStreamDescSize(stream_desc, static_cast<uint32_t>(raw_pkt->size));
        // EOS=0: explicitly mark each frame as not end-of-stream.
        // Some CANN6 driver builds treat the default (unset) as EOS, causing stalls.
        acldvppSetStreamDescEos(stream_desc, 0);

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

        fctx->slot_yuv_buf  = slot->yuv_buf;
        fctx->in_pic_desc   = slot->pic_desc;
        fctx->vpc_enabled   = vpc_enabled_;
        fctx->vpc_out_w     = vpc_out_w_;
        fctx->vpc_out_h     = vpc_out_h_;
        fctx->vpc_scaler    = vpc_scaler_ptr;
        fctx->vpc_pool      = vpc_output_pool;
        fctx->acl_context   = ctx_handle;

        auto pool_sp = output_pool;
        fctx->pool_release = std::shared_ptr<void>(
            reinterpret_cast<void*>(1),
            [pool_sp, slot](void*) { pool_sp->release(slot); }
        );

        aclError send_rc = ACL_ERROR_FAILURE;
#if CANN_VERSION_MAJOR >= 7
        auto vdec = dvpp_api_.vdecProcess ? dvpp_api_.vdecProcess
            : VdecProcessFn([](AclVdecChannelDesc* ch, acldvppStreamDesc* sd,
                 acldvppPicDesc* pd, aclrtStream s,
                 AclVdecCb fn, void* ud) -> aclError {
                return acldvppVdecProcess(ch, sd, pd, s, fn, ud);
            });
        send_rc = vdec(channel_desc_, stream_desc, slot->pic_desc, dvpp_stream_,
                       &DVPPDecoder::onDecoded, fctx);
#else
        auto vdec = dvpp_api_.vdecProcess ? dvpp_api_.vdecProcess
            : VdecProcessFn([](AclVdecChannelDesc* ch, acldvppStreamDesc* sd,
                 acldvppPicDesc* pd, aclvdecFrameConfig* cfg, void* ud) -> aclError {
                return aclvdecSendFrame(ch, sd, pd, cfg, ud);
            });
        // session_frame_cfg is shared across all frames in this channel session.
        // EOS is signalled via acldvppSetStreamDescEos on the stream descriptor (above).
        send_rc = vdec(channel_desc_, stream_desc, slot->pic_desc, session_frame_cfg, fctx);
#endif
        if (send_rc != ACL_SUCCESS) {
            LOG_ERROR("DVPPDecoder: sendFrame failed (rc={})",
                      static_cast<int>(send_rc));
            // On failure DVPP will NOT invoke the callback — we own all resources.
            acldvppDestroyStreamDesc(stream_desc);
            acldvppFree(bitstream_dev);
            delete fctx;
            output_pool->release(slot);
            return false;
        }
        // On success: stream_desc is owned by DVPP until the callback fires.
        //   onDecoded() destroys stream_desc via its `input` parameter.
        //   slot is returned to the pool via fctx->pool_release when Frame is released.
        return true;
    };

    // ── Packet decode loop ────────────────────────────────────────────────
    // On 5 consecutive sendFrame failures the channel is poisoned; break out
    // so the outer reconnect loop can tear down and recreate everything.
    //
    // Keyframe-first: DVPP cannot decode P/B frames without a prior reference
    // frame. If non-IDR frames are submitted before the first IDR, DVPP
    // silently discards them WITHOUT calling the decode callback. Since each
    // submitted frame holds an output pool slot, a stream that starts mid-GOP
    // will exhaust the pool instantly. Skip all packets until the first IDR.
    AVPacket* pkt = av_packet_alloc();
    int consecutive_failures = 0;
    constexpr int kMaxConsecutiveFailures = 5;
    bool channel_poisoned = false;
    bool got_keyframe = false;

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
                if (!got_keyframe) {
                    if (pkt->flags & AV_PKT_FLAG_KEY) {
                        got_keyframe = true;
                        LOG_INFO("DVPPDecoder [{}]: first keyframe seen, starting submission",
                                 stream_id_);
                    } else {
                        av_packet_unref(pkt);
                        continue;
                    }
                }
                if (submitToDvpp(pkt)) send_ok = true;
                av_packet_unref(pkt);
            }
        } else {
            if (!got_keyframe) {
                if (pkt->flags & AV_PKT_FLAG_KEY) {
                    got_keyframe = true;
                    LOG_INFO("DVPPDecoder [{}]: first keyframe seen, starting submission",
                             stream_id_);
                } else {
                    av_packet_unref(pkt);
                    continue;
                }
            }
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

        // Pump pending DVPP callbacks for this thread's report queue.
        // CANN 6 vdec dispatches the decode-complete callback to the thread
        // registered via aclvdecSetChannelDescThreadId() (set in initChannel,
        // which is this same decodeLoop thread). Without pumping here the
        // queue is never drained and onDecoded never fires.
        // Use 5ms timeout: long enough to actually catch a callback between
        // packets (av_read_frame at ~30ms/frame), short enough to keep the
        // submit loop responsive. ACL_ERROR_RT_REPORT_TIMEOUT is the expected
        // benign return when no callback is ready and is intentionally ignored.
#if CANN_VERSION_MAJOR < 7
        (void)aclrtProcessReport(5);
#endif
    }

    // ── Per-connection cleanup ────────────────────────────────────────────
    // No aclrtUnSubscribeReport — we never called aclrtSubscribeReport for
    // vdec; the thread binding is owned by the channel desc and released by
    // aclvdecDestroyChannel.
    av_packet_free(&pkt);
    if (bsf_ctx) av_bsf_free(&bsf_ctx);
    avformat_close_input(&fmt_ctx);
    // aclvdecDestroyChannel blocks until all in-flight callbacks complete (CANN6 guarantee).
    // Destroy channel first so no more callbacks can fire before we reset the pool.
    destroyChannel();
    // session_frame_cfg must outlive all in-flight aclvdecSendFrame calls.
    // Safe to destroy now: destroyChannel() above has drained all pending callbacks.
#if CANN_VERSION_MAJOR < 7
    if (session_frame_cfg) { aclvdecDestroyFrameConfig(session_frame_cfg); session_frame_cfg = nullptr; }
#endif
    if (vpc_output_pool) {
        vpc_output_pool->destroy();
        vpc_output_pool.reset();
    }
    if (vpc_enabled_) {
        vpc_scaler_ptr->shutdown();
    }

    output_pool->destroy();
    output_pool.reset();

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
