#include "pipeline/stages/SahiRoiRegistry.h"

#include <mutex>
#include <unordered_map>

namespace infer {

namespace {

std::mutex& registryMutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, SahiRoiSnapshot>& registryData() {
    static std::unordered_map<std::string, SahiRoiSnapshot> data;
    return data;
}

} // namespace

void SahiRoiRegistry::update(const std::string& stream_id, uint64_t frame_seq, const std::vector<Detection>& detections) {
    std::lock_guard<std::mutex> lock(registryMutex());
    registryData()[stream_id] = SahiRoiSnapshot{frame_seq, detections};
}

std::optional<SahiRoiSnapshot> SahiRoiRegistry::get(const std::string& stream_id) {
    std::lock_guard<std::mutex> lock(registryMutex());
    auto it = registryData().find(stream_id);
    if (it == registryData().end()) return std::nullopt;
    return it->second;
}

} // namespace infer
