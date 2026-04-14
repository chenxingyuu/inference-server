#ifdef BUILD_TRT_BACKEND

#include "infer/TRTBackend.h"
#include "common/Logger.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace infer {

namespace {

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            LOG_WARN("[TRT] {}", msg);
    }
} g_trt_logger;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("[CUDA] ") + cudaGetErrorString(err)); \
    } \
} while(0)

} // namespace

TRTBackend::TRTBackend() = default;

TRTBackend::~TRTBackend() {
    unloadModel();
}

void TRTBackend::loadModel(const ModelConfig& cfg) {
    device_id_      = cfg.device_id;
    max_batch_size_ = cfg.batch_size;
    input_h_        = cfg.input_shape.height;
    input_w_        = cfg.input_shape.width;
    num_classes_    = cfg.num_classes;

    CUDA_CHECK(cudaSetDevice(device_id_));

    // Read engine file
    std::ifstream file(cfg.engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("TRTBackend: cannot open engine: " + cfg.engine_path);
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    if (!engine_) {
        throw std::runtime_error("TRTBackend: deserializeCudaEngine failed");
    }
    context_.reset(engine_->createExecutionContext());

    allocateBuffers(max_batch_size_);
    loaded_ = true;
    LOG_INFO("TRTBackend: loaded engine {} (batch={})", cfg.engine_path, max_batch_size_);
}

void TRTBackend::allocateBuffers(int batch_size) {
    // input: [batch, 3, H, W]  float32
    input_size_  = batch_size * 3 * input_h_ * input_w_ * sizeof(float);
    // output size depends on model; allocate generously
    // YOLOv8 output: [batch, 4+num_classes, num_anchors] ≈ [batch, 84, 8400]
    output_size_ = batch_size * (4 + num_classes_) * 8400 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&device_buffers_[0], input_size_));
    CUDA_CHECK(cudaMalloc(&device_buffers_[1], output_size_));

    input_staging_.resize(input_size_ / sizeof(float));
    output_staging_.resize(output_size_ / sizeof(float));
}

void TRTBackend::preprocessCPU(const Batch& input, float* dst,
                                int batch_size, int h, int w) {
    const float inv_255 = 1.0f / 255.0f;
    for (int b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(input.frames[b], resized, {w, h});
        // Convert BGR uint8 → CHW float [0,1]
        float* ch_r = dst + b * 3 * h * w;
        float* ch_g = ch_r + h * w;
        float* ch_b = ch_g + h * w;
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                ch_b[y * w + x] = row[x * 3 + 0] * inv_255;
                ch_g[y * w + x] = row[x * 3 + 1] * inv_255;
                ch_r[y * w + x] = row[x * 3 + 2] * inv_255;
            }
        }
    }
}

void TRTBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) throw std::runtime_error("TRTBackend: model not loaded");
    const int bs = input.size();
    if (bs > max_batch_size_) {
        throw std::runtime_error("TRTBackend: batch size exceeds max");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));

    preprocessCPU(input, input_staging_.data(), bs, input_h_, input_w_);

    size_t in_bytes = bs * 3 * input_h_ * input_w_ * sizeof(float);
    CUDA_CHECK(cudaMemcpy(device_buffers_[0], input_staging_.data(),
                          in_bytes, cudaMemcpyHostToDevice));

    // Set dynamic batch size if supported
    context_->setInputShape("images",
        nvinfer1::Dims4{bs, 3, input_h_, input_w_});

    bool ok = context_->executeV2(device_buffers_);
    if (!ok) throw std::runtime_error("TRTBackend: executeV2 failed");

    size_t out_bytes = bs * (4 + num_classes_) * 8400 * sizeof(float);
    output.resize(out_bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(output.data(), device_buffers_[1],
                          out_bytes, cudaMemcpyDeviceToHost));
}

void TRTBackend::unloadModel() {
    if (device_buffers_[0]) { cudaFree(device_buffers_[0]); device_buffers_[0] = nullptr; }
    if (device_buffers_[1]) { cudaFree(device_buffers_[1]); device_buffers_[1] = nullptr; }
    context_.reset();
    engine_.reset();
    runtime_.reset();
    loaded_ = false;
}

} // namespace infer

#endif // BUILD_TRT_BACKEND
