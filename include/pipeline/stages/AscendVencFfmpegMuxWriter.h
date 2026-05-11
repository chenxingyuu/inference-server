#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include "pipeline/stages/DrawAndStreamStage.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace infer {

// DVPP VENC (CANN 6 aclvenc*) → Annex-B H.264 → ffmpeg stdin (-f h264 -c:v copy) → RTSP/RTMP.
class AscendVencFfmpegMuxWriter final : public IStreamWriter {
public:
    explicit AscendVencFfmpegMuxWriter(int device_id);
    ~AscendVencFfmpegMuxWriter() override;

    bool open(const std::string& url,
                const std::string& protocol,
                double fps,
                int gop,
                int bitrate_kbps,
                int width,
                int height) override;
    bool write(const cv::Mat& frame) override;
    void close() override;
    bool isOpened() const override;

private:
    static void onVencCallback(acldvppPicDesc* input, acldvppStreamDesc* output, void* userdata);

    void destroyEncoder();
    void rollbackOpen();
    void closeFfmpegPipe();

    int device_id_{0};
    FILE* pipe_{nullptr};
    aclrtContext ctx_{nullptr};
    aclrtStream stream_{nullptr};

    aclvencChannelDesc* channel_desc_{nullptr};
    bool                venc_channel_created_{false};
    void*               venc_out_buf_{nullptr};
    uint32_t            venc_out_buf_size_{0};
    bool                runtime_acquired_{false};

    uint32_t pic_w_{0};
    uint32_t pic_h_{0};
    size_t   nv12_host_bytes_{0};
    std::vector<uint8_t> nv12_host_;

    std::mutex          pipe_mu_;
    std::atomic<bool>   opened_{false};
    std::atomic<bool>   frame_done_{false};
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
