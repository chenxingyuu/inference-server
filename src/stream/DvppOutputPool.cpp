#ifdef BUILD_ASCEND_BACKEND

#include "stream/DvppOutputPool.h"

#include <chrono>

namespace infer {

bool DvppOutputPool::init(uint32_t ys, uint32_t cw, uint32_t ch, uint32_t aw, uint32_t ah) {
    yuv_size = ys;
    codec_w  = cw;
    codec_h  = ch;
    aligned_w = aw;
    aligned_h = ah;
    for (auto& s : slots) {
        if (acldvppMalloc(&s.yuv_buf, yuv_size) != ACL_SUCCESS || !s.yuv_buf)
            return false;
        s.pic_desc = acldvppCreatePicDesc();
        if (!s.pic_desc) return false;
        resetDesc(s);
    }
    return true;
}

void DvppOutputPool::resetDesc(Slot& s) {
    acldvppSetPicDescData(s.pic_desc, s.yuv_buf);
    acldvppSetPicDescSize(s.pic_desc, yuv_size);
    acldvppSetPicDescFormat(s.pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(s.pic_desc, codec_w);
    acldvppSetPicDescHeight(s.pic_desc, codec_h);
    acldvppSetPicDescWidthStride(s.pic_desc, aligned_w);
    acldvppSetPicDescHeightStride(s.pic_desc, aligned_h);
}

DvppOutputPool::Slot* DvppOutputPool::acquire(int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
        for (auto& s : slots) {
            if (!s.in_use) return true;
        }
        return false;
    });
    for (auto& s : slots) {
        if (!s.in_use) {
            s.in_use = true;
            return &s;
        }
    }
    return nullptr;
}

void DvppOutputPool::release(Slot* s) {
    resetDesc(*s);
    {
        std::lock_guard<std::mutex> lk(mu);
        s->in_use = false;
    }
    cv.notify_one();
}

void DvppOutputPool::destroy() {
    for (auto& s : slots) {
        if (s.pic_desc) {
            acldvppDestroyPicDesc(s.pic_desc);
            s.pic_desc = nullptr;
        }
        if (s.yuv_buf) {
            acldvppFree(s.yuv_buf);
            s.yuv_buf = nullptr;
        }
        s.in_use = false;
    }
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
