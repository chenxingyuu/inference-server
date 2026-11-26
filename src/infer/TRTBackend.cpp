#ifdef BUILD_TRT_BACKEND

#include "infer/TRTBackend.h"
#include "infer/GpuFault.h"
#include "cuda/CudaPreprocess.h"
#include "common/Logger.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <limits>
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
    if (input_h_ <= 0 || input_w_ <= 0 || input_h_ > 4096 || input_w_ > 4096) {
        throw std::runtime_error("TRTBackend: input resolution must be in range (0, 4096]");
    }

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
    LOG_DEBUG("TRTBackend: tensors resolved input={} output={}",
              input_tensor_name_, output_tensor_name_);
    const nvinfer1::Dims in_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    if (in_dims.nbDims <= 0) {
        throw std::runtime_error(
            "TRTBackend: input tensor not found in engine: " + input_tensor_name_);
    }
    input_shape_dynamic_ = hasDynamicDims(in_dims.nbDims, in_dims.d);
    LOG_DEBUG("TRTBackend: input shape dynamic={} nbDims={}", input_shape_dynamic_, in_dims.nbDims);

    if (io_tensors != 2 || in_dims.nbDims != 4 ||
        engine_->getTensorDataType(input_tensor_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
        engine_->getTensorDataType(output_tensor_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
        engine_->getTensorFormat(input_tensor_name_.c_str()) != nvinfer1::TensorFormat::kLINEAR ||
        engine_->getTensorFormat(output_tensor_name_.c_str()) != nvinfer1::TensorFormat::kLINEAR ||
        engine_->getTensorLocation(input_tensor_name_.c_str()) != nvinfer1::TensorLocation::kDEVICE ||
        engine_->getTensorLocation(output_tensor_name_.c_str()) != nvinfer1::TensorLocation::kDEVICE)
        throw std::invalid_argument("TensorRT model must have one linear FP32 input and output");

    const auto min_shape = input_shape_dynamic_
        ? engine_->getProfileShape(input_tensor_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMIN) : in_dims;
    const auto max_shape = input_shape_dynamic_
        ? engine_->getProfileShape(input_tensor_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX) : in_dims;
    if (min_shape.nbDims != 4 || max_shape.nbDims != 4 || min_shape.d[0] < 1 ||
        max_shape.d[0] < min_shape.d[0] || max_shape.d[0] > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("TensorRT model has an invalid batch profile");
    }
    min_batch_size_ = static_cast<int>(min_shape.d[0]);
    max_batch_size_ = std::clamp(max_batch_size_, min_batch_size_, static_cast<int>(max_shape.d[0]));

    const size_t frame_bytes = static_cast<size_t>(3) * input_h_ * input_w_ * sizeof(float);
    if (static_cast<size_t>(max_batch_size_) > std::numeric_limits<size_t>::max() / frame_bytes)
        throw std::runtime_error("TensorRT input is too large");
    input_bytes_ = static_cast<size_t>(max_batch_size_) * frame_bytes;
    output_bytes_ = resolveOutputBytes(max_batch_size_);

    // Initialize pre-allocated GPU buffer pool (Phase 7a-2)
    const int pool_size = (cfg.buffer_pool_size > 0) ? cfg.buffer_pool_size : 4;
    buffer_pool_.init(device_id_, pool_size, input_bytes_, output_bytes_);

    // CPU-path staging buffers
    input_staging_.resize(input_bytes_ / sizeof(float));

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
    LOG_INFO("TRTBackend: loaded engine={} device={} batch={} input={}x{}x{}x{} pool={}",
             cfg.engine_path, device_id_, max_batch_size_, max_batch_size_, 3, input_h_, input_w_, pool_size);
}

size_t TRTBackend::resolveOutputBytes(int batch_size) {
    const nvinfer1::Dims4 shape{batch_size, 3, input_h_, input_w_};
    if (!context_->setInputShape(input_tensor_name_.c_str(), shape)) {
        throw std::invalid_argument("TensorRT model does not accept the configured input shape");
    }
    const auto output_shape = context_->getTensorShape(output_tensor_name_.c_str());
    size_t bytes = sizeof(float);
    if (output_shape.nbDims < 1)
        throw std::invalid_argument("TensorRT output shape could not be resolved");
    for (int i = 0; i < output_shape.nbDims; ++i) {
        const auto dimension = output_shape.d[i];
        if (dimension < 1 || static_cast<size_t>(dimension) > std::numeric_limits<size_t>::max() / bytes)
            throw std::invalid_argument("TensorRT output shape is unsupported");
        bytes *= static_cast<size_t>(dimension);
    }
    return bytes;
}

void TRTBackend::bindTensors(PooledBuffer* slot) {
    if (!context_->setTensorAddress(input_tensor_name_.c_str(), slot->input_device) ||
        !context_->setTensorAddress(output_tensor_name_.c_str(), slot->output_device)) {
        throw std::runtime_error("TensorRT could not bind inference buffers");
    }
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

    const auto out_bytes = resolveOutputBytes(bs);
    if (out_bytes > output_bytes_) throw std::runtime_error("TensorRT output exceeds the allocated buffer");

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

        bindTensors(slot);
        if (!context_->enqueueV3(infer_stream_))
            throw std::runtime_error("TensorRT inference failed (GPU path)");

        // Copy the actual output tensor back on the inference stream.
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
    if (bs < min_batch_size_ || bs > max_batch_size_) {
        throw std::runtime_error("TRTBackend: batch size is outside the engine profile");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));

    const auto infer_start = std::chrono::steady_clock::now();

    if (input.is_gpu) {
        LOG_DEBUG("TRTBackend: GPU infer start bs={}", bs);
        inferGPU(input, output);
        checkInferTimeout(infer_start);
        return;
    }

    if (input.frames.size() != static_cast<size_t>(bs))
        throw std::runtime_error("TRTBackend: CPU frames do not match the batch");
    const auto out_bytes = resolveOutputBytes(bs);
    if (out_bytes > output_bytes_) throw std::runtime_error("TensorRT output exceeds the allocated buffer");

    // CPU path: host preprocessing → H2D → inference → D2H (synchronous)
    LOG_WARN("TRTBackend: using CPU preprocess path (is_gpu=false, hwdec not active) bs={}", bs);
    auto pre_start = std::chrono::steady_clock::now();
    preprocessCPU(input, input_staging_.data(), bs, input_h_, input_w_);
    const double pre_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pre_start).count();
    LOG_WARN("TRTBackend: CPU preprocess done bs={} pre_ms={:.1f}", bs, pre_ms);

    PooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) {
        throw std::runtime_error("TRTBackend: GpuBufferPool exhausted (CPU path)");
    }

    try {
        size_t in_bytes = bs * 3 * input_h_ * input_w_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(slot->input_device, input_staging_.data(),
                              in_bytes, cudaMemcpyHostToDevice));

        bindTensors(slot);
        if (!context_->enqueueV3(infer_stream_))
            throw std::runtime_error("TensorRT inference failed (CPU input)");
        CUDA_CHECK(cudaStreamSynchronize(infer_stream_));

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
    LOG_INFO("TRTBackend: unloading model (device={})", device_id_);
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
