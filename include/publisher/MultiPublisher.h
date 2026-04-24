#pragma once

#include "publisher/IPublisher.h"
#include <vector>
#include <memory>
#include <cstddef>

namespace infer {

// Fan-out publisher: forwards publish()/flush() to every child publisher.
// Exceptions from individual children are caught and logged; the remaining
// children are still called so one bad publisher cannot starve others.
// Thread-safety: each child must be individually thread-safe; this class
// adds no additional synchronization.
class MultiPublisher final : public IPublisher {
public:
    // Throws std::invalid_argument if publishers is empty.
    explicit MultiPublisher(std::vector<std::unique_ptr<IPublisher>> publishers);

    void publish(InferResult result) override;
    void flush() override;

    std::size_t size() const noexcept { return publishers_.size(); }

private:
    std::vector<std::unique_ptr<IPublisher>> publishers_;
};

} // namespace infer
