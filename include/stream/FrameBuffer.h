#pragma once

#include "common/Types.h"
#include <boost/lockfree/spsc_queue.hpp>
#include <atomic>

namespace infer {

// Lock-free single-producer / single-consumer ring buffer for decoded frames.
// Producer: StreamDecoder thread.  Consumer: BatchScheduler thread.
// Capacity is fixed at compile time; oldest frames are dropped when full.
static constexpr std::size_t kFrameBufferCapacity = 32;

class FrameBuffer {
public:
    explicit FrameBuffer(const std::string& stream_id)
        : stream_id_(stream_id) {}

    // Non-copyable / non-movable (boost::lockfree queue is not movable)
    FrameBuffer(const FrameBuffer&)            = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;

    // Returns false if the buffer was full (frame dropped)
    bool push(Frame frame);

    // Returns false if the buffer was empty
    bool pop(Frame& out);

    // NOTE: boost::lockfree::spsc_queue::empty() is non-const in Boost < 1.76
    bool empty() noexcept { return queue_.empty(); }
    const std::string& streamId() const noexcept { return stream_id_; }

    uint64_t droppedFrames() const noexcept { return dropped_.load(std::memory_order_relaxed); }

private:
    std::string stream_id_;
    boost::lockfree::spsc_queue<Frame,
        boost::lockfree::capacity<kFrameBufferCapacity>> queue_;
    std::atomic<uint64_t> dropped_{0};
};

} // namespace infer
