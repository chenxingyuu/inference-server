#pragma once

#ifdef BUILD_TRT_BACKEND

#include "infer/IInferBackend.h"
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
    void allocateBuffers(int batch_size);
    void preprocessCPU(const Batch& input, float* dst, int batch_size,
                       int h, int w);
    void preprocessGPU(const Batch& input, int batch_size);
    void inferGPU(const Batch& input, std::vector<float>& output);

    int  max_batch_size_{1};
    int  input_h_{640};
    int  input_w_{640};
    int  num_classes_{80};
    bool loaded_{false};
    int  device_id_{0};

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>        engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    // Device buffers [0] = input, [1] = output
    void* device_buffers_[2]{nullptr, nullptr};
    std::vector<float> input_staging_;
    std::vector<float> output_staging_;

    size_t input_size_{0};
    size_t output_size_{0};

    cudaStream_t infer_stream_{nullptr};
};

} // namespace infer

#endif // BUILD_TRT_BACKEND
