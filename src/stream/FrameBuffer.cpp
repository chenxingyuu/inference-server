#include "stream/FrameBuffer.h"

namespace infer {

bool FrameBuffer::push(Frame frame) {
    bool ok = queue_.push(std::move(frame));
    if (!ok) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

bool FrameBuffer::pop(Frame& out) {
    return queue_.pop(out);
}

} // namespace infer
