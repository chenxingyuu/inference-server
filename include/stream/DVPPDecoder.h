#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include "stream/IStreamDecoder.h"
#include "common/Types.h"
#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

namespace infer {

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
    using CreateChannelFn  = std::function<aclError(acldvppChannelDesc**)>;
    using DestroyChannelFn = std::function<aclError(acldvppChannelDesc*)>;
    using VdecProcessFn    = std::function<
        aclError(acldvppChannelDesc*, acldvppStreamDesc*,
                 acldvppPicDesc*, aclrtStream, aclDvppVdecCallback, void*)>;

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
    bool initChannelForTest(int device_id) { return initChannel(device_id); }
    void destroyChannelForTest()            { destroyChannel(); }

    // Expose the DVPP frame callback for direct unit-test invocation.
    // In tests: call onDecoded(nullptr, fake_pic_desc, user_data) to verify
    // that a Frame with is_ascend=true is constructed and forwarded.
    static void onDecoded(acldvppStreamDesc* input,
                          acldvppPicDesc*    output,
                          void*              user_data);

private:
    void decodeLoop(StreamConfig cfg, FrameCallback cb);
    bool initChannel(int device_id);
    void destroyChannel();

    std::string              stream_id_;
    std::thread              thread_;
    std::atomic<bool>        running_{false};
    std::atomic<bool>        stop_flag_{false};
    int                      device_id_{0};
    aclrtStream              dvpp_stream_{nullptr};
    acldvppChannelDesc*      channel_desc_{nullptr};

    DvppApiStub              dvpp_api_{};  // zero-initialized → real ACL path

    // Context passed to DVPP callback
    struct CallbackCtx {
        FrameCallback cb;
        int           device_id{0};
    };
    std::mutex               ctx_mu_;
    CallbackCtx              ctx_;

    static constexpr int kOutputPoolSize = 4;
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
