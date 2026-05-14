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

// ── Output buffer pool ─────────────────────────────────────────────────────
//
// CANN 6 requirement (hiascend.com/developer/techArticles/20231215-1):
//   When a frame decode fails, DVPP resets the output acldvppPicDesc parameters
//   to 0 (format=0 → YUV400, which VDEC doesn't support). All subsequent frames
//   then fail with error 507018.
//
// Fix: pre-allocate a fixed pool of output slots. Before each aclvdecSendFrame,
// explicitly reset format/width/height/stride on the slot's acldvppPicDesc.
// On callback, the slot is returned to the pool (not freed) and reset again.
// This bounds DVPP memory and prevents descriptor parameter corruption.
struct OutputPool {
    struct Slot {
        void*           yuv_buf{nullptr};
        acldvppPicDesc* pic_desc{nullptr};
        bool            in_use{false};
    };

    static constexpr int kSize = 8;
    std::array<Slot, kSize> slots{};
    std::mutex              mu;
    std::condition_variable cv;

    uint32_t yuv_size{0};
    uint32_t codec_w{0}, codec_h{0};
    uint32_t aligned_w{0}, aligned_h{0};

    bool init(uint32_t ys, uint32_t cw, uint32_t ch, uint32_t aw, uint32_t ah) {
        yuv_size = ys; codec_w = cw; codec_h = ch; aligned_w = aw; aligned_h = ah;
        for (auto& s : slots) {
            if (acldvppMalloc(&s.yuv_buf, yuv_size) != ACL_SUCCESS || !s.yuv_buf)
                return false;
            s.pic_desc = acldvppCreatePicDesc();
            if (!s.pic_desc) return false;
            resetDesc(s);
        }
        return true;
    }

    void resetDesc(Slot& s) {
        acldvppSetPicDescData(s.pic_desc, s.yuv_buf);
        acldvppSetPicDescSize(s.pic_desc, yuv_size);
        acldvppSetPicDescFormat(s.pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
        acldvppSetPicDescWidth(s.pic_desc, codec_w);
        acldvppSetPicDescHeight(s.pic_desc, codec_h);
        acldvppSetPicDescWidthStride(s.pic_desc, aligned_w);
        acldvppSetPicDescHeightStride(s.pic_desc, aligned_h);
    }

    // Block until a slot is available (up to timeout_ms). Returns nullptr on timeout.
    Slot* acquire(int timeout_ms = 200) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
            for (auto& s : slots) if (!s.in_use) return true;
            return false;
        });
        for (auto& s : slots) {
            if (!s.in_use) { s.in_use = true; return &s; }
        }
        return nullptr;
    }

    // Return slot to pool; reset desc params (mandatory per CANN 6 docs).
    void release(Slot* s) {
        resetDesc(*s);
        {
            std::lock_guard<std::mutex> lk(mu);
            s->in_use = false;
        }
        cv.notify_one();
    }

    void destroy() {
        for (auto& s : slots) {
            if (s.pic_desc) { acldvppDestroyPicDesc(s.pic_desc); s.pic_desc = nullptr; }
            if (s.yuv_buf)  { acldvppFree(s.yuv_buf);            s.yuv_buf  = nullptr; }
            s.in_use = false;
        }
    }
};

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

    // Codec type: must exactly match the actual bitstream profile.
    // CANN6 silently drops frames (no callback) when the profile doesn't match.
    // Surveillance cameras almost universally use H.264 High Profile (idc=100+).
    // H264_HIGH_LEVEL is a superset and safely decodes Main/Baseline streams too.
    // acldvppStreamFormat: H265_MAIN_LEVEL=0, H264_BASELINE_LEVEL=1,
    //                       H264_MAIN_LEVEL=2, H264_HIGH_LEVEL=3.
    const acldvppStreamFormat en_type = is_h265 ? H265_MAIN_LEVEL : H264_HIGH_LEVEL;
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
//   - output pic_desc + yuv_buf: owned by the OutputPool slot; NOT freed here.
//     frame_ref deleter calls pool_release (returns slot to pool and resets desc).
//
// user_data = heap-allocated FrameCtx* (deleted here after callback returns).

void DVPPDecoder::onDecoded(acldvppStreamDesc* input,
                             acldvppPicDesc*    output,
                             void*              user_data) {
    if (!user_data) return;
    auto* fctx = static_cast<FrameCtx*>(user_data);
    LOG_INFO("DVPPDecoder [{}]: onDecoded fired seq={} output={}",
             fctx->stream_id, fctx->frame_seq, output ? "ok" : "null");

    // Destroy the stream descriptor — DVPP passes it back here after decode completes.
    if (input) acldvppDestroyStreamDesc(input);

    // Release the compressed bitstream device memory.
    if (fctx->bitstream_dev) {
        acldvppFree(fctx->bitstream_dev);
        fctx->bitstream_dev = nullptr;
    }

    AscendBuffer buf;
    buf.device_id      = fctx->device_id;
    buf.aligned_width  = static_cast<int>(fctx->aligned_width);
    buf.aligned_height = static_cast<int>(fctx->aligned_height);

    // output != nullptr means DVPP wrote decoded YUV into slot->yuv_buf.
    // slot_yuv_buf is the pool slot's device memory; pool_release returns the
    // slot to the pool (and resets its desc params) when the Frame is released.
    // We do NOT call acldvppFree or acldvppDestroyPicDesc here — the pool owns them.
    if (output && fctx->slot_yuv_buf) {
        buf.yuv_device = fctx->slot_yuv_buf;
        buf.width      = fctx->codec_width;
        buf.height     = fctx->codec_height;
        buf.frame_ref  = std::move(fctx->pool_release);
    }

    const double decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - fctx->submit_ts).count();
    LOG_DEBUG("DVPPDecoder: hw decode done {}x{} decode_ms={:.1f}",
              buf.width, buf.height, decode_ms);

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

    // ── Pre-allocate output buffer pool ──────────────────────────────────
    // CANN 6 requires output acldvppPicDesc params to be explicitly reset before
    // each aclvdecSendFrame. Reusing pool slots (rather than malloc/free per frame)
    // ensures params are always reset and bounds DVPP memory usage.
    const uint32_t yuv_size = alignedYuvSize(codec_width_, codec_height_);
    auto output_pool = std::make_shared<OutputPool>();
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
             stream_id_, OutputPool::kSize, yuv_size);

    // ── Single-packet DVPP submit helper ─────────────────────────────────
    //
    // CANN6 aclvdecSendFrame is ASYNCHRONOUS: the call returns as soon as the
    // frame is queued; the decode callback fires on a DVPP worker thread when
    // the hardware finishes. Lifetime rules:
    //   - bitstream_dev: freed inside onDecoded (per-frame allocation)
    //   - yuv_buf / pic_desc: owned by the output pool; returned via frame_ref deleter
    auto submitToDvpp = [&](AVPacket* raw_pkt) -> bool {
        // Acquire a free output slot (blocks up to 200 ms if pool is full).
        OutputPool::Slot* slot = output_pool->acquire(200);
        if (!slot) {
            LOG_WARN("DVPPDecoder [{}]: output pool exhausted, dropping frame", stream_id_);
            return true;  // not a channel error — treat as soft drop
        }
        // Always reset desc params before use (mandatory per CANN 6 docs).
        output_pool->resetDesc(*slot);

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

        // pool_release: returned to pool (with desc reset) when the Frame is released.
        fctx->slot_yuv_buf = slot->yuv_buf;
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
        aclvdecFrameConfig* frame_cfg = aclvdecCreateFrameConfig();
        if (!frame_cfg) {
            LOG_ERROR("DVPPDecoder: aclvdecCreateFrameConfig returned null");
            delete fctx;
            acldvppFree(bitstream_dev);
            acldvppDestroyStreamDesc(stream_desc);
            output_pool->release(slot);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }
        // Explicitly mark as NOT end-of-stream. The default may be 1, which
        // signals EOS to DVPP and prevents the callback from firing.
        aclvdecSetFrameConfigEos(frame_cfg, 0);
        send_rc = vdec(channel_desc_, stream_desc, slot->pic_desc, frame_cfg, fctx);
        aclvdecDestroyFrameConfig(frame_cfg);
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
    }

    // ── Per-connection cleanup ────────────────────────────────────────────
    av_packet_free(&pkt);
    if (bsf_ctx) av_bsf_free(&bsf_ctx);
    avformat_close_input(&fmt_ctx);
    // aclvdecDestroyChannel blocks until all in-flight callbacks complete (CANN6 guarantee).
    // Destroy channel first so no more callbacks can fire before we reset the pool.
    destroyChannel();
    // Reset pool ownership here. Any frames still in the downstream pipeline keep the
    // shared_ptr alive; their frame_ref deleter will call pool->release() safely.
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
