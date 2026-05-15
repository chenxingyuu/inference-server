#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include "stream/IStreamDecoder.h"
#include "common/Types.h"
#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace infer {

// CANN 6 and CANN 7+ have fundamentally different vdec APIs:
//   CANN ≤ 6: aclvdecChannelDesc, callback set on descriptor, aclvdecSendFrame()
//   CANN ≥ 7: acldvppChannelDesc, callback passed per-call, acldvppVdecProcess()
#if CANN_VERSION_MAJOR >= 7
using AclVdecCb          = aclDvppVdecCallback;
using AclVdecChannelDesc = acldvppChannelDesc;
#else
using AclVdecCb          = aclvdecCallback;
using AclVdecChannelDesc = aclvdecChannelDesc;
#endif

// DVPP hardware video decoder for Ascend 310P.
//
// Input:  H.264/H.265 RTSP bitstream (demuxed by FFmpeg avformat — no avcodec decode)
// Output: YUV420SP (NV12) frames in NPU HBM device memory → Frame.ascend_buf
//
// When the associated model was compiled with AIPP, decoded frames bypass CPU
// entirely: DVPP output → AIPP (in .om) → NPU inference → Kafka result.
//
// Implements IStreamDecoder so SourceRtspStage can select between
// FFmpegDecoder and DVPPDecoder via StreamConfig.use_ascend_dvpp.
class DVPPDecoder : public IStreamDecoder {
public:
    // ── Injectable DVPP API stubs (for unit tests without real NPU) ───────
    using CreateChannelFn  = std::function<aclError(AclVdecChannelDesc**)>;
    using DestroyChannelFn = std::function<aclError(AclVdecChannelDesc*)>;
#if CANN_VERSION_MAJOR >= 7
    using VdecProcessFn    = std::function<
        aclError(AclVdecChannelDesc*, acldvppStreamDesc*,
                 acldvppPicDesc*, aclrtStream, AclVdecCb, void*)>;
#else
    // CANN 6: callback is registered on the channel descriptor; userData passed per-send.
    using VdecProcessFn    = std::function<
        aclError(AclVdecChannelDesc*, acldvppStreamDesc*,
                 acldvppPicDesc*, aclvdecFrameConfig*, void*)>;
#endif

    struct DvppApiStub {
        CreateChannelFn  createChannel;
        DestroyChannelFn destroyChannel;
        VdecProcessFn    vdecProcess;
    };

    DVPPDecoder();
    ~DVPPDecoder() override;

    // IStreamDecoder interface
    void start(const StreamConfig& cfg, FrameCallback cb) override;
    void stop()                                           override;
    bool running()             const override { return running_.load(); }
    const std::string& streamId() const override { return stream_id_; }

    // Inject DVPP API stubs for testing (call before start())
    void setDvppApiForTest(DvppApiStub stubs) { dvpp_api_ = std::move(stubs); }

    // Start a lightweight test loop that does NOT open RTSP or call DVPP.
    // Returns once the loop thread has started and running() == true.
    void startForTest(FrameCallback cb);

    // Call initChannel() / destroyChannel() directly (uses injected dvpp_api_).
    bool initChannelForTest(int device_id, uint32_t aligned_w = 640,
                            uint32_t aligned_h = 640, bool is_h265 = false) {
        return initChannel(device_id, aligned_w, aligned_h, is_h265);
    }
    void destroyChannelForTest() { destroyChannel(); }

    // Return the per-instance channel ID assigned at construction.
    uint32_t channelIdForTest() const { return channel_id_; }

    // Expose the DVPP frame callback for direct unit-test invocation.
    static void onDecoded(acldvppStreamDesc* input,
                          acldvppPicDesc*    output,
                          void*              user_data);

    // Compute DVPP-aligned YUV420SP buffer size.
    // DVPP requires: width stride aligned to 16, height stride aligned to 2.
    static uint32_t alignedYuvSize(uint32_t w, uint32_t h) {
        const uint32_t aw = (w + 15u) & ~15u;
        const uint32_t ah = (h +  1u) & ~1u;
        return aw * ah * 3u / 2u;
    }

    // Per-frame context heap-allocated for each aclvdecSendFrame call.
    // Carries everything onDecoded() needs to build a complete Frame with meta.
    // Exposed in public so unit tests can heap-allocate and pass to onDecoded().
    struct FrameCtx {
        FrameCallback  cb;
        int            device_id{0};
        void*          bitstream_dev{nullptr};
        std::chrono::steady_clock::time_point submit_ts;
        std::string    stream_id;
        uint64_t       frame_seq{0};
        uint32_t       aligned_width{0};
        uint32_t       aligned_height{0};
        int            codec_width{0};
        int            codec_height{0};
        double         capture_ts{0.0};      // epoch seconds at submission
        uint64_t       capture_mono_ns{0};   // steady_clock ns at submission

        // Output buffer pool slot management.
        // pool_release: shared_ptr whose deleter returns the slot to the pool.
        // slot_yuv_buf: device pointer inside the slot (written by DVPP, read by NPU).
        std::shared_ptr<void> pool_release;
        void*                 slot_yuv_buf{nullptr};
    };

private:
    void decodeLoop(StreamConfig cfg);
    bool initChannel(int device_id, uint32_t aligned_w, uint32_t aligned_h,
                     int h264_profile, bool is_h265 = false);
    void destroyChannel();

    std::string              stream_id_;
    std::thread              thread_;
    std::atomic<bool>        running_{false};
    std::atomic<bool>        stop_flag_{false};
    int                      device_id_{0};
    aclrtStream              dvpp_stream_{nullptr};
    AclVdecChannelDesc*      channel_desc_{nullptr};

    // Per-instance unique channel ID (avoids DVPP channel conflicts on multi-stream).
    uint32_t                 channel_id_{0};

    // Stream geometry — set in decodeLoop after avformat_find_stream_info.
    uint32_t                 codec_width_{0};
    uint32_t                 codec_height_{0};
    uint32_t                 aligned_width_{0};
    uint32_t                 aligned_height_{0};

    DvppApiStub              dvpp_api_{};

    // Persistent per-instance context (cb + device_id); set in start(), read by decodeLoop.
    struct CallbackCtx {
        FrameCallback cb;
        int           device_id{0};
    };
    std::mutex               ctx_mu_;
    CallbackCtx              ctx_;

    // Startup synchronization: decodeLoop signals once it knows it will produce
    // frames (startup_ok_=true) or has given up (startup_ok_=false).
    // Lets start() wait for the actual result instead of a fixed timeout.
    std::mutex               startup_mu_;
    std::condition_variable  startup_cv_;
    bool                     startup_done_{false};
    bool                     startup_ok_{false};

    // Process-wide channel counter — each DVPPDecoder gets a unique channel ID.
    static std::atomic<uint32_t> s_channel_counter_;

    uint64_t                 frame_seq_{0};   // monotonic per-stream frame counter

    static constexpr int kOutputPoolSize = 32;
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
