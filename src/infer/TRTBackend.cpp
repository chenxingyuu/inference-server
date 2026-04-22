#ifdef BUILD_TRT_BACKEND

#include "infer/TRTBackend.h"
#include "infer/GpuFault.h"
#include "cuda/CudaPreprocess.h"
#include "common/Logger.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "infer/TrtUtils.h"

namespace infer {

namespace {

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            LOG_WARN("[TRT] {}", msg);
    }
} g_trt_logger;

// Classify and throw a GpuFaultException for any non-success CUDA error.
#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw GpuFaultException(classifyGpuError(static_cast<int>(_err)), \
                                cudaGetErrorString(_err)); \
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
    if (!context_) {
        throw std::runtime_error("TRTBackend: createExecutionContext failed");
    }

    // Cache tensor names and whether input dims are dynamic.
    const int io_tensors = engine_->getNbIOTensors();
    input_tensor_name_.clear();
    output_tensor_name_.clear();
    for (int i = 0; i < io_tensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (!name) continue;
        const auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && input_tensor_name_.empty())
            input_tensor_name_ = name;
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_tensor_name_.empty())
            output_tensor_name_ = name;
    }
    if (input_tensor_name_.empty()) {
        throw std::runtime_error("TRTBackend: unable to resolve input tensor name");
    }
    if (output_tensor_name_.empty()) {
        throw std::runtime_error("TRTBackend: unable to resolve output tensor name");
    }
    const nvinfer1::Dims in_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    if (in_dims.nbDims <= 0) {
        throw std::runtime_error(
            "TRTBackend: input tensor not found in engine: " + input_tensor_name_);
    }
    input_shape_dynamic_ = hasDynamicDims(in_dims.nbDims, in_dims.d);

    // Pre-compute buffer sizes
    input_bytes_  = max_batch_size_ * 3 * input_h_ * input_w_ * sizeof(float);
    output_bytes_ = max_batch_size_ * (4 + num_classes_) * 8400 * sizeof(float);

    // Initialize pre-allocated GPU buffer pool (Phase 7a-2)
    const int pool_size = (cfg.buffer_pool_size > 0) ? cfg.buffer_pool_size : 4;
    buffer_pool_.init(device_id_, pool_size, input_bytes_, output_bytes_);

    // CPU-path staging buffers
    input_staging_.resize(input_bytes_ / sizeof(float));
    output_staging_.resize(output_bytes_ / sizeof(float));

    // Two CUDA streams for the async double-buffered pipeline (Phase 7a-1).
    // preprocess_stream_: NV12→CHW kernel
    // infer_stream_:      TRT enqueueV3 + D2H copy
    CUDA_CHECK(cudaStreamCreate(&preprocess_stream_));
    CUDA_CHECK(cudaStreamCreate(&infer_stream_));

    // Event that signals the preprocessing kernel is complete.
    // infer_stream_ waits on it before starting inference.
    CUDA_CHECK(cudaEventCreateWithFlags(&preprocess_done_,
                                        cudaEventDisableTiming));

    loaded_ = true;
    LOG_INFO("TRTBackend: loaded engine {} (batch={}, pool={})",
             cfg.engine_path, max_batch_size_, pool_size);
}

void TRTBackend::preprocessCPU(const Batch& input, float* dst,
                                int batch_size, int h, int w) {
    const float inv_255 = 1.0f / 255.0f;
    for (int b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(input.frames[b], resized, {w, h});
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

void TRTBackend::preprocessGPU(const Batch& input,
                                PooledBuffer* slot, int batch_size) {
    // Stage 1: Launch NV12→CHW fused CUDA kernel on preprocess_stream_.
    // Each frame's NV12 buffer is written into the pre-allocated input slot.
    // After all frames are processed, record preprocess_done_ so infer_stream_
    // can wait on it without blocking the host.
    const float mean  = 0.0f;
    const float scale = 1.0f / 255.0f;
    const int   plane = input_h_ * input_w_;

    for (int b = 0; b < batch_size; ++b) {
        const GpuBuffer& gb = input.gpu_frames[b];
        float* dst_offset   = static_cast<float*>(slot->input_device) + b * 3 * plane;

        cuda::preprocessNV12ToCHW(
            static_cast<const uint8_t*>(gb.y_data),
            static_cast<const uint8_t*>(gb.uv_data),
            gb.height, gb.width,
            dst_offset, input_h_, input_w_,
            mean, scale,
            preprocess_stream_);
    }

    // Signal that all input pixels are written.
    CUDA_CHECK(cudaEventRecord(preprocess_done_, preprocess_stream_));
}

void TRTBackend::inferGPU(const Batch& input, std::vector<float>& output) {
    const int bs = input.size();

    if (static_cast<int>(input.gpu_frames.size()) != bs) {
        throw std::runtime_error(
            "TRTBackend: gpu_frames.size() != metas.size() — heterogeneous batch");
    }

    // Acquire a pre-allocated buffer slot from the pool (Phase 7a-2).
    PooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) {
        throw std::runtime_error("TRTBackend: GpuBufferPool exhausted");
    }

    try {
        // Stage 1: async preprocessing on preprocess_stream_ (Phase 7a-1).
        preprocessGPU(input, slot, bs);

        // Stage 2: infer_stream_ waits for preprocessing without blocking the host.
        // This allows preprocess_stream_ to overlap with a future batch's
        // setup work while infer_stream_ is scheduling TRT work.
        CUDA_CHECK(cudaStreamWaitEvent(infer_stream_, preprocess_done_, 0));

        // Always set input shape: no-op on static engines, required for dynamic.
        context_->setInputShape(input_tensor_name_.c_str(),
                                nvinfer1::Dims4{bs, 3, input_h_, input_w_});

        // Use enqueueV3 (TRT 8.5+) for full async dispatch on infer_stream_.
        // Falls back to executeV2 if enqueueV3 is not available at link time.
#if NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR >= 5
        context_->setTensorAddress(input_tensor_name_.c_str(), slot->input_device);
        context_->setTensorAddress(output_tensor_name_.c_str(), slot->output_device);
        bool ok = context_->enqueueV3(infer_stream_);
        if (!ok) throw std::runtime_error("TRTBackend: enqueueV3 failed");
#else
        void* bindings[2] = {slot->input_device, slot->output_device};
        bool ok = context_->executeV2(bindings);
        if (!ok) throw std::runtime_error("TRTBackend: executeV2 failed (GPU path)");
#endif

        // Stage 3: async D2H copy on infer_stream_ (follows inference in stream order).
        size_t out_bytes = bs * (4 + num_classes_) * 8400 * sizeof(float);
        output.resize(out_bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(output.data(), slot->output_device,
                                   out_bytes, cudaMemcpyDeviceToHost, infer_stream_));

        // Block host until D2H copy and inference are complete.
        // The preprocess_stream_ is free to start the next batch's kernel
        // while this sync is in progress (overlap achieved via pool spare slots).
        CUDA_CHECK(cudaStreamSynchronize(infer_stream_));
    } catch (...) {
        buffer_pool_.release(slot);
        throw;
    }

    buffer_pool_.release(slot);
}

void TRTBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) throw std::runtime_error("TRTBackend: model not loaded");
    const int bs = input.size();
    if (bs > max_batch_size_) {
        throw std::runtime_error("TRTBackend: batch size exceeds max");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));

    const auto infer_start = std::chrono::steady_clock::now();

    if (input.is_gpu) {
        inferGPU(input, output);
        checkInferTimeout(infer_start);
        return;
    }

    // CPU path: host preprocessing → H2D → inference → D2H (synchronous)
    preprocessCPU(input, input_staging_.data(), bs, input_h_, input_w_);

    PooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) {
        throw std::runtime_error("TRTBackend: GpuBufferPool exhausted (CPU path)");
    }

    try {
        size_t in_bytes = bs * 3 * input_h_ * input_w_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(slot->input_device, input_staging_.data(),
                              in_bytes, cudaMemcpyHostToDevice));

        // Always set input shape: no-op on static engines, required for dynamic.
        context_->setInputShape(input_tensor_name_.c_str(),
                                nvinfer1::Dims4{bs, 3, input_h_, input_w_});
        context_->setTensorAddress(input_tensor_name_.c_str(),  slot->input_device);
        context_->setTensorAddress(output_tensor_name_.c_str(), slot->output_device);
        void* bindings[2] = {slot->input_device, slot->output_device};
        bool ok = context_->executeV2(bindings);
        if (!ok) throw std::runtime_error("TRTBackend: executeV2 failed");

        size_t out_bytes = bs * (4 + num_classes_) * 8400 * sizeof(float);
        output.resize(out_bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpy(output.data(), slot->output_device,
                              out_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        buffer_pool_.release(slot);
        throw;
    }

    buffer_pool_.release(slot);
    checkInferTimeout(infer_start);
}

void TRTBackend::checkInferTimeout(
    const std::chrono::steady_clock::time_point& start) const {
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    LOG_DEBUG("TRTBackend: infer latency_ms={:.1f}", elapsed_ms);
    if (elapsed_ms > infer_timeout_ms_) {
        throw GpuFaultException(GpuFaultType::TIMEOUT,
            "TRTBackend: inference exceeded " +
            std::to_string(static_cast<int>(infer_timeout_ms_)) + "ms timeout");
    }
}

void TRTBackend::unloadModel() {
    if (!loaded_) return;
    cudaSetDevice(device_id_);
    if (preprocess_done_)   { cudaEventDestroy(preprocess_done_);   preprocess_done_   = nullptr; }
    if (preprocess_stream_) { cudaStreamDestroy(preprocess_stream_); preprocess_stream_ = nullptr; }
    if (infer_stream_)      { cudaStreamDestroy(infer_stream_);      infer_stream_      = nullptr; }
    buffer_pool_.reset();
    context_.reset();
    engine_.reset();
    runtime_.reset();
    loaded_ = false;
}

} // namespace infer

#endif // BUILD_TRT_BACKEND
