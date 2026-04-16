#include "infer/BackendFactory.h"
#include "infer/TRTBackend.h"
#include "infer/AscendBackend.h"
#ifdef BUILD_ONNX_BACKEND
#include "infer/OnnxBackend.h"
#endif
#include <stdexcept>

namespace infer {

std::unique_ptr<IInferBackend> createBackend(const ModelConfig& cfg) {
    switch (cfg.backend) {
    case DeviceType::CUDA:
#ifdef BUILD_TRT_BACKEND
        return std::make_unique<TRTBackend>();
#else
        throw std::runtime_error("TensorRT backend not compiled in "
                                 "(rebuild with BUILD_TRT_BACKEND=ON)");
#endif
    case DeviceType::Ascend:
#ifdef BUILD_ASCEND_BACKEND
        return std::make_unique<AscendBackend>();
#else
        throw std::runtime_error("Ascend backend not compiled in "
                                 "(rebuild with BUILD_ASCEND_BACKEND=ON)");
#endif
    case DeviceType::CPU:
    case DeviceType::MPS:
#ifdef BUILD_ONNX_BACKEND
        return std::make_unique<OnnxBackend>(cfg.backend);
#else
        throw std::runtime_error("ONNX backend not compiled in "
                                 "(rebuild with BUILD_ONNX_BACKEND=ON)");
#endif
    }
    throw std::runtime_error("Unknown backend type");
}

} // namespace infer
