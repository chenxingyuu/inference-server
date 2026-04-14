#include "infer/BackendFactory.h"
#include "infer/TRTBackend.h"
#include "infer/AscendBackend.h"
#include <stdexcept>

namespace infer {

std::unique_ptr<IInferBackend> createBackend(const ModelConfig& cfg) {
    switch (cfg.backend) {
    case DeviceType::CUDA:
#ifdef BUILD_TRT_BACKEND
        return std::make_unique<TRTBackend>();
#else
        throw std::runtime_error("TensorRT backend not compiled in");
#endif
    case DeviceType::Ascend:
#ifdef BUILD_ASCEND_BACKEND
        return std::make_unique<AscendBackend>();
#else
        throw std::runtime_error("Ascend backend not compiled in");
#endif
    default:
        throw std::runtime_error("Unknown backend type");
    }
}

} // namespace infer
