#pragma once

#include <stdexcept>
#include <string>

namespace infer {

enum class GpuFaultType {
    OOM,           // cudaErrorMemoryAllocation — out of GPU memory
    ENGINE_FAULT,  // TRT inference failure or unclassified CUDA error
    CONTEXT_LOST,  // cudaErrorInvalidContext / cudaErrorInvalidDevice
    TIMEOUT,       // inference watchdog expired
};

// Thrown by TRTBackend when a CUDA call fails or inference times out.
struct GpuFaultException : std::runtime_error {
    GpuFaultType fault_type;
    explicit GpuFaultException(GpuFaultType t, const std::string& msg)
        : std::runtime_error(msg), fault_type(t) {}
};

// Thrown by GpuBufferPool::acquire() when memory usage exceeds the safety threshold.
// The engine has NOT crashed — no reload needed, just back-pressure.
struct GpuMemoryPressureException : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// CUDA error code constants (mirrors cuda_runtime_api.h values).
// Defined here to avoid requiring CUDA headers in non-GPU translation units.
inline constexpr int kCudaSuccess             = 0;
inline constexpr int kCudaErrorMemAlloc       = 2;   // cudaErrorMemoryAllocation
inline constexpr int kCudaErrorInvalidContext = 201; // cudaErrorInvalidContext
inline constexpr int kCudaErrorInvalidDevice  = 101; // cudaErrorInvalidDevice

// Map a raw CUDA error code to a GpuFaultType.
// Takes int so it can be used without CUDA headers (pass static_cast<int>(cudaError_t)).
inline GpuFaultType classifyGpuError(int cuda_err_code) {
    if (cuda_err_code == kCudaErrorMemAlloc)
        return GpuFaultType::OOM;
    if (cuda_err_code == kCudaErrorInvalidContext ||
        cuda_err_code == kCudaErrorInvalidDevice)
        return GpuFaultType::CONTEXT_LOST;
    return GpuFaultType::ENGINE_FAULT;
}

} // namespace infer
