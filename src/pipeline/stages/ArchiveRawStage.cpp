#include "pipeline/stages/ArchiveRawStage.h"

#include "common/Logger.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace infer {
namespace {

bool toArchivableMat(const Frame& frame, cv::Mat& out) {
    if (!frame.is_gpu) {
        if (frame.image.empty()) return false;
        out = frame.image;
        return true;
    }

    // Graceful fallback for tests or mixed pipelines where a CPU copy exists.
    if (!frame.image.empty()) {
        out = frame.image;
        return true;
    }

    AVFrame* hw_frame = static_cast<AVFrame*>(frame.gpu_buf.frame_ref.get());
    if (!hw_frame) return false;

    AVFrame* sw_frame = av_frame_alloc();
    if (!sw_frame) return false;

    const int transfer_rc = av_hwframe_transfer_data(sw_frame, hw_frame, 0);
    if (transfer_rc < 0) {
        av_frame_free(&sw_frame);
        return false;
    }
    const auto sw_format = static_cast<AVPixelFormat>(sw_frame->format);
    if (sw_format != AV_PIX_FMT_NV12 && sw_format != AV_PIX_FMT_YUV420P) {
        LOG_WARN("ArchiveRawStage: unsupported software pixel format={} for GPU archive",
                 static_cast<int>(sw_format));
        av_frame_free(&sw_frame);
        return false;
    }

    SwsContext* sws_ctx = sws_getContext(
        sw_frame->width, sw_frame->height, sw_format,
        sw_frame->width, sw_frame->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        av_frame_free(&sw_frame);
        return false;
    }

    cv::Mat bgr(sw_frame->height, sw_frame->width, CV_8UC3);
    uint8_t* dst_data[1] = {bgr.data};
    int dst_linesize[1] = {static_cast<int>(bgr.step)};
    sws_scale(sws_ctx, sw_frame->data, sw_frame->linesize, 0, sw_frame->height, dst_data, dst_linesize);

    sws_freeContext(sws_ctx);
    av_frame_free(&sw_frame);
    out = std::move(bgr);
    return true;
}

} // namespace

ArchiveRawStage::ArchiveRawStage(std::string id, std::shared_ptr<FrameArchiver> archiver, bool allow_gpu_frames)
    : id_(std::move(id)), archiver_(std::move(archiver)), allow_gpu_frames_(allow_gpu_frames) {}

std::string ArchiveRawStage::id() const { return id_; }

void ArchiveRawStage::process(const EventEnvelope& input, const EmitFn& emit) {
    EventEnvelope out = input;
    cv::Mat archivable;
    const bool should_attempt_archive = !input.frame || !input.frame->is_gpu || allow_gpu_frames_;
    if (archiver_ && input.frame && should_attempt_archive && toArchivableMat(*input.frame, archivable)) {
        auto result = archiver_->submit(input.frame->meta, &archivable);
        out.archive_info = ArchiveInfo{result.local_path, result.upload_state};
    } else {
        if (input.frame && input.frame->is_gpu && archiver_ && allow_gpu_frames_) {
            LOG_WARN("ArchiveRawStage: unable to archive GPU frame stream={} seq={}",
                     input.frame->meta.stream_id,
                     input.frame->meta.frame_seq);
        }
        // Archive disabled/unavailable: set disabled so JoinByFrameStage is not blocked.
        out.archive_info = ArchiveInfo{"", "disabled"};
    }
    emit(out);
}

} // namespace infer
