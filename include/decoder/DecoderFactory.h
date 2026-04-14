#pragma once

#include "decoder/IYOLODecoder.h"
#include "common/Config.h"
#include <memory>

namespace infer {

// Create the correct YOLO decoder for the given ModelConfig.
std::unique_ptr<IYOLODecoder> createDecoder(const ModelConfig& cfg);

} // namespace infer
