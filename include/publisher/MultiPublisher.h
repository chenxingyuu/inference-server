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
    // Owning: takes exclusive ownership of each child publisher.
    // Throws std::invalid_argument if publishers is empty.
    explicit MultiPublisher(std::vector<std::unique_ptr<IPublisher>> publishers);

    // Non-owning: borrows raw pointers from an external registry.
    // Callers must ensure all pointed-to publishers outlive this object.
    // Throws std::invalid_argument if publishers is empty.
    explicit MultiPublisher(std::vector<IPublisher*> publishers);

    void publish(InferResult result) override;
    void flush() override;

    std::size_t size() const noexcept { return refs_.size(); }

private:
    std::vector<std::unique_ptr<IPublisher>> owned_;  // non-empty for owning variant
    std::vector<IPublisher*>                 refs_;   // always valid (points into owned_ or external)
};

} // namespace infer
