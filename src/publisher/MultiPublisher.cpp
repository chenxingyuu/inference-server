#include "publisher/MultiPublisher.h"
#include "common/Logger.h"
#include <stdexcept>

namespace infer {

MultiPublisher::MultiPublisher(std::vector<std::unique_ptr<IPublisher>> publishers)
    : publishers_(std::move(publishers)) {
    if (publishers_.empty()) {
        throw std::invalid_argument("MultiPublisher: at least one publisher is required");
    }
}

void MultiPublisher::publish(InferResult result) {
    for (auto& pub : publishers_) {
        try {
            pub->publish(result);
        } catch (const std::exception& e) {
            LOG_ERROR("MultiPublisher: child publish() failed: {}", e.what());
        }
    }
}

void MultiPublisher::flush() {
    for (auto& pub : publishers_) {
        try {
            pub->flush();
        } catch (const std::exception& e) {
            LOG_ERROR("MultiPublisher: child flush() failed: {}", e.what());
        }
    }
}

} // namespace infer
