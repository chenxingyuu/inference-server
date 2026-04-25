#pragma once

#include <mutex>
#include <vector>

namespace infer {

// Thread-safe least-loaded GPU device allocator.
// Streams call acquire() on start to get the GPU with fewest active streams,
// and release() on stop to free the slot.
class GpuDeviceAllocator {
public:
    explicit GpuDeviceAllocator(std::vector<int> device_ids);

    // Returns device_id of the GPU with the fewest active streams.
    // Thread-safe.
    int acquire();

    // Decrements active stream count for device_id. Count never goes below 0.
    // Thread-safe.
    void release(int device_id);

    // Returns current active stream count for device_id.
    // Thread-safe.
    int activeStreams(int device_id) const;

private:
    mutable std::mutex mu_;
    std::vector<std::pair<int, int>> devices_; // {device_id, active_count}
};

} // namespace infer
