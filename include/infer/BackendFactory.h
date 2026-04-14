#pragma once

#include "infer/IInferBackend.h"
#include "common/Config.h"
#include <memory>

namespace infer {

// Create the correct backend based on ModelConfig::backend.
// Throws std::runtime_error if the requested backend was not compiled in.
std::unique_ptr<IInferBackend> createBackend(const ModelConfig& cfg);

} // namespace infer
