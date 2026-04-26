#ifdef BUILD_ASCEND_BACKEND

#include "stream/DVPPDecoder.h"
#include "common/Logger.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <stdexcept>
#include <chrono>

// FFmpeg headers (demux only — no avcodec decode)
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#define ACL_DVPP_CHECK(call) do { \
    aclError _e = (call); \
    if (_e != ACL_SUCCESS) { \
        LOG_WARN("DVPPDecoder ACL error {} at {}:{}", _e, __FILE__, __LINE__); \
    } \
} while(0)

namespace infer {

// ── Lifecycle ──────────────────────────────────────────────────────────────

DVPPDecoder::DVPPDecoder() = default;

DVPPDecoder::~DVPPDecoder() {
    stop();
}

void DVPPDecoder::start(const StreamConfig& cfg, FrameCallback cb) {
    stream_id_  = cfg.id;
    device_id_  = cfg.ascend_device_id;
    stop_flag_.store(false, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(ctx_mu_);
        ctx_.cb        = std::move(cb);
        ctx_.device_id = device_id_;
    }

    thread_ = std::thread(&DVPPDecoder::decodeLoop, this, cfg,
                          [this](Frame f) {
                              std::lock_guard<std::mutex> lk(ctx_mu_);
                              if (ctx_.cb) ctx_.cb(std::move(f));
                          });
    // Spin until running_ is confirmed true (set inside decodeLoop)
    for (int i = 0; i < 1000 && !running_.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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

bool DVPPDecoder::initChannel(int device_id) {
    // Create DVPP vdec channel descriptor
    acldvppChannelDesc* desc = acldvppCreateChannelDesc();
    if (!desc) {
        LOG_ERROR("DVPPDecoder: acldvppCreateChannelDesc() returned null");
        return false;
    }
    channel_desc_ = desc;

    // Create ACL stream for DVPP async operations
    aclError err = aclrtCreateStream(&dvpp_stream_);
    if (err != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: aclrtCreateStream failed ({})", static_cast<int>(err));
        return false;
    }

    // Use injected stub if set (test path), otherwise call real DVPP
    aclError create_err;
    if (dvpp_api_.createChannel) {
        create_err = dvpp_api_.createChannel(&channel_desc_);
    } else {
        create_err = acldvppCreateChannel(channel_desc_);
    }

    if (create_err != ACL_SUCCESS) {
        LOG_ERROR("DVPPDecoder: acldvppCreateChannel failed ({})",
                  static_cast<int>(create_err));
        return false;
    }

    LOG_INFO("DVPPDecoder: DVPP vdec channel created, device={}", device_id);
    return true;
}

void DVPPDecoder::destroyChannel() {
    if (channel_desc_) {
        aclError err;
        if (dvpp_api_.destroyChannel) {
            err = dvpp_api_.destroyChannel(channel_desc_);
        } else {
            err = acldvppDestroyChannel(channel_desc_);
        }
        if (err != ACL_SUCCESS) {
            LOG_WARN("DVPPDecoder: acldvppDestroyChannel failed ({})",
                     static_cast<int>(err));
        }
        acldvppDestroyChannelDesc(channel_desc_);
        channel_desc_ = nullptr;
    }
    if (dvpp_stream_) {
        aclrtDestroyStream(dvpp_stream_);
        dvpp_stream_ = nullptr;
    }
}

// ── DVPP decode callback ────────────────────────────────────────────────────

void DVPPDecoder::onDecoded(acldvppStreamDesc* /*input*/,
                             acldvppPicDesc*    output,
                             void*              user_data) {
    if (!user_data) return;
    auto* ctx = static_cast<CallbackCtx*>(user_data);

    // Build AscendBuffer from DVPP output descriptor
    AscendBuffer buf;
    buf.device_id = ctx->device_id;

    if (output) {
        buf.yuv_device = acldvppGetPicDescData(output);
        buf.width      = static_cast<int>(acldvppGetPicDescWidth(output));
        buf.height     = static_cast<int>(acldvppGetPicDescHeight(output));

        // frame_ref deleter returns the buffer to DVPP pool
        acldvppPicDesc* desc_copy = output;
        buf.frame_ref = std::shared_ptr<void>(
            reinterpret_cast<void*>(1),  // non-null sentinel
            [desc_copy](void*) { acldvppDestroyPicDesc(desc_copy); }
        );
    }
    // output == nullptr is valid in test mode; buf.yuv_device remains null.

    Frame f;
    f.ascend_buf = std::move(buf);
    f.is_ascend  = true;

    if (ctx->cb) ctx->cb(std::move(f));
}

// ── Decode loop ────────────────────────────────────────────────────────────

void DVPPDecoder::decodeLoop(StreamConfig cfg, FrameCallback cb) {
    // ACL context must be created in the thread that will use it
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

    // Lock-in the user callback into our context struct
    {
        std::lock_guard<std::mutex> lk(ctx_mu_);
        ctx_.cb = std::move(cb);
    }

    if (!initChannel(device_id_)) {
        aclrtDestroyContext(ctx_handle);
        return;
    }

    running_.store(true, std::memory_order_release);

    // ── FFmpeg demux loop ─────────────────────────────────────────────────
    AVFormatContext* fmt_ctx = nullptr;
    AVDictionary*    opts    = nullptr;
    // Timeout option (microseconds) to avoid blocking forever on bad URLs
    av_dict_set(&opts, "stimeout", "5000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);

    if (avformat_open_input(&fmt_ctx, cfg.url.c_str(), nullptr, &opts) < 0) {
        LOG_ERROR("DVPPDecoder: cannot open RTSP: {}", cfg.url);
        av_dict_free(&opts);
        destroyChannel();
        running_.store(false, std::memory_order_release);
        aclrtDestroyContext(ctx_handle);
        return;
    }
    av_dict_free(&opts);
    avformat_find_stream_info(fmt_ctx, nullptr);

    // Find video stream index
    int video_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = static_cast<int>(i);
            break;
        }
    }
    if (video_idx < 0) {
        LOG_ERROR("DVPPDecoder: no video stream in {}", cfg.url);
        avformat_close_input(&fmt_ctx);
        destroyChannel();
        running_.store(false, std::memory_order_release);
        aclrtDestroyContext(ctx_handle);
        return;
    }

    AVPacket* pkt = av_packet_alloc();
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

        // Build DVPP stream descriptor from compressed packet
        acldvppStreamDesc* stream_desc = acldvppCreateStreamDesc();
        if (stream_desc) {
            acldvppSetStreamDescData(stream_desc, pkt->data);
            acldvppSetStreamDescSize(stream_desc, static_cast<uint32_t>(pkt->size));

            // Pre-allocate output pic descriptor (DVPP writes decoded YUV here)
            acldvppPicDesc* pic_desc = acldvppCreatePicDesc();
            if (pic_desc) {
                void* yuv_buf = nullptr;
                // YUV420SP: H * W * 3 / 2 bytes
                const uint32_t yuv_size = static_cast<uint32_t>(
                    fmt_ctx->streams[video_idx]->codecpar->height
                    * fmt_ctx->streams[video_idx]->codecpar->width * 3 / 2);
                acldvppMalloc(&yuv_buf, yuv_size);
                acldvppSetPicDescData(pic_desc, yuv_buf);
                acldvppSetPicDescSize(pic_desc, yuv_size);

                auto vdec = dvpp_api_.vdecProcess ? dvpp_api_.vdecProcess
                    : [](acldvppChannelDesc* ch, acldvppStreamDesc* sd,
                         acldvppPicDesc* pd, aclrtStream s,
                         aclDvppVdecCallback fn, void* ud) -> aclError {
                        return acldvppVdecProcess(ch, sd, pd, s, fn, ud);
                    };

                // The callback takes ownership of pic_desc and frees it on completion
                vdec(channel_desc_, stream_desc, pic_desc, dvpp_stream_,
                     &DVPPDecoder::onDecoded, &ctx_);
            }
            acldvppDestroyStreamDesc(stream_desc);
        }

        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);
    avformat_close_input(&fmt_ctx);
    destroyChannel();
    running_.store(false, std::memory_order_release);
    aclrtDestroyContext(ctx_handle);
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
