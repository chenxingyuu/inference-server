#pragma once

#ifdef BUILD_TRT_BACKEND

#include "infer/IInferBackend.h"
#include "infer/GpuBufferPool.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>

// Forward declarations to avoid pulling NvInfer.h into every TU
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace infer {

// TensorRT inference backend with async double-buffered pipeline.
//
// Execution stages (inspired by DeepStream's full-async pipeline):
//
//   Stage 1 (preprocess_stream_):
//     CUDA kernel: NV12 → CHW float, resize, normalize → slot->input_device
//     cudaEventRecord(preprocess_done_)
//
//   Stage 2 (infer_stream_):
//     cudaStreamWaitEvent(infer_stream_, preprocess_done_)
//     enqueueV3() — TRT 8.5+ full-async inference
//     cudaEventRecord(infer_done_)
//
//   Stage 3 (caller thread):
//     cudaStreamWaitEvent(infer_stream_, infer_done_) or StreamSync
//     async D2H memcpy → output vector
//
// For the CPU-path fallback, stages collapse to the legacy synchronous flow.
class TRTBackend : public IInferBackend {
public:
    TRTBackend();
    ~TRTBackend() override;

    void loadModel(const ModelConfig& cfg) override;
    void infer(const Batch& input, std::vector<float>& output) override;
    void unloadModel() override;

    int        maxBatchSize() const override { return max_batch_size_; }
    DeviceType deviceType()   const override { return DeviceType::CUDA; }
    bool       isLoaded()     const override { return loaded_; }

private:
    void preprocessCPU(const Batch& input, float* dst, int batch_size,
                       int h, int w);
    // Async GPU preprocess: launches kernel on preprocess_stream_, records event.
    void preprocessGPU(const Batch& input, PooledBuffer* slot, int batch_size);
    // Full async inference: waits for preprocess event, runs enqueueV3 on infer_stream_.
    void inferGPU(const Batch& input, std::vector<float>& output);

    int  max_batch_size_{1};
    int  input_h_{640};
    int  input_w_{640};
    int  num_classes_{80};
    bool loaded_{false};
    int  device_id_{0};
    bool input_shape_dynamic_{false};
    std::string input_tensor_name_{"images"};
    std::string output_tensor_name_;

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>        engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    // Pre-allocated GPU buffer pool (replaces per-inference cudaMalloc)
    GpuBufferPool buffer_pool_;
    size_t input_bytes_{0};
    size_t output_bytes_{0};

    // Staging buffers for CPU inference path
    std::vector<float> input_staging_;
    std::vector<float> output_staging_;

    // Two CUDA streams: preprocessing runs on preprocess_stream_,
    // TRT inference runs on infer_stream_.  CUDA Events synchronize them
    // without blocking the host, enabling stage overlap when the pool has
    // spare slots (controlled by buffer_pool_size).
    cudaStream_t preprocess_stream_{nullptr};
    cudaStream_t infer_stream_{nullptr};

    // Event signalling that preprocessing has written the input buffer.
    // infer_stream_ waits on this before calling enqueueV3.
    cudaEvent_t preprocess_done_{nullptr};
};

} // namespace infer

#endif // BUILD_TRT_BACKEND
