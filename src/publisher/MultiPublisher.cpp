#include "publisher/MultiPublisher.h"
#include "common/Logger.h"
#include <stdexcept>

namespace infer {

MultiPublisher::MultiPublisher(std::vector<std::unique_ptr<IPublisher>> publishers)
    : owned_(std::move(publishers)) {
    if (owned_.empty())
        throw std::invalid_argument("MultiPublisher: at least one publisher is required");
    refs_.reserve(owned_.size());
    for (auto& p : owned_) refs_.push_back(p.get());
}

MultiPublisher::MultiPublisher(std::vector<IPublisher*> publishers)
    : refs_(std::move(publishers)) {
    if (refs_.empty())
        throw std::invalid_argument("MultiPublisher: at least one publisher is required");
}

void MultiPublisher::publish(InferResult result) {
    for (auto* pub : refs_) {
        try {
            pub->publish(result);
        } catch (const std::exception& e) {
            LOG_ERROR("MultiPublisher: child publish() failed: {}", e.what());
        }
    }
}

void MultiPublisher::flush() {
    for (auto* pub : refs_) {
        try {
            pub->flush();
        } catch (const std::exception& e) {
            LOG_ERROR("MultiPublisher: child flush() failed: {}", e.what());
        }
    }
}

} // namespace infer
