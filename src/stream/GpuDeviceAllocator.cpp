#include "stream/GpuDeviceAllocator.h"

#include <algorithm>

namespace infer {

GpuDeviceAllocator::GpuDeviceAllocator(std::vector<int> device_ids) {
    if (device_ids.empty()) {
        throw std::invalid_argument("GpuDeviceAllocator: device_ids must not be empty");
    }
    devices_.reserve(device_ids.size());
    for (int id : device_ids) {
        devices_.push_back({id, 0});
    }
}

int GpuDeviceAllocator::acquire() {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::min_element(devices_.begin(), devices_.end(),
                               [](const auto& a, const auto& b) { return a.second < b.second; });
    it->second++;
    return it->first;
}

void GpuDeviceAllocator::release(int device_id) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [id, count] : devices_) {
        if (id == device_id) {
            if (count > 0) count--;
            return;
        }
    }
}

int GpuDeviceAllocator::activeStreams(int device_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& [id, count] : devices_) {
        if (id == device_id) return count;
    }
    return 0;
}

} // namespace infer
