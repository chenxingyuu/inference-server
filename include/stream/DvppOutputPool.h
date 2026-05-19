#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include <acl/ops/acl_dvpp.h>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace infer {

// Fixed-size pool of DVPP YUV420SP output buffers + pic descriptors (CANN 6 safe reuse).
struct DvppOutputPool {
    struct Slot {
        void*           yuv_buf{nullptr};
        acldvppPicDesc* pic_desc{nullptr};
        bool            in_use{false};
    };

    static constexpr int kSize = 32;
    std::array<Slot, kSize> slots{};
    std::mutex              mu;
    std::condition_variable cv;

    uint32_t yuv_size{0};
    uint32_t codec_w{0}, codec_h{0};
    uint32_t aligned_w{0}, aligned_h{0};

    bool init(uint32_t ys, uint32_t cw, uint32_t ch, uint32_t aw, uint32_t ah);
    void resetDesc(Slot& s);
    Slot* acquire(int timeout_ms = 200);
    void release(Slot* s);
    void destroy();
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
