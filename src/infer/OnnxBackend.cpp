#include "infer/OnnxBackend.h"
#include "common/Logger.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <thread>

#ifdef BUILD_ONNX_BACKEND_COREML
#include <coreml_provider_factory.h>
#endif

namespace infer {

OnnxBackend::OnnxBackend(DeviceType device)
    : device_(device)
    , env_(ORT_LOGGING_LEVEL_WARNING, "OnnxBackend")
    , session_opts_()
{}

OnnxBackend::~OnnxBackend() {
    if (loaded_) {
        unloadModel();
    }
}

void OnnxBackend::loadModel(const ModelConfig& cfg) {
    if (loaded_) {
        throw std::runtime_error("OnnxBackend::loadModel called while model already loaded");
    }
    if (cfg.onnx_path.empty()) {
        throw std::runtime_error("OnnxBackend: onnx_path must be set in model config");
    }

    max_batch_size_ = std::max(1, cfg.batch_size);
    num_classes_    = cfg.num_classes;

    // ── Session options ────────────────────────────────────────────────────────
    session_opts_.SetIntraOpNumThreads(
        static_cast<int>(std::thread::hardware_concurrency()));
    session_opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // ── Execution provider ────────────────────────────────────────────────────
    if (device_ == DeviceType::MPS) {
#ifdef BUILD_ONNX_BACKEND_COREML
        uint32_t coreml_flags = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
            session_opts_, coreml_flags));
        LOG_INFO("OnnxBackend: CoreML (MPS) execution provider enabled");
#else
        LOG_WARN("OnnxBackend: MPS requested but CoreML EP not compiled "
                 "(rebuild with BUILD_ONNX_BACKEND_COREML=ON); falling back to CPU");
#endif
    }

    // ── Create session ────────────────────────────────────────────────────────
    session_ = std::make_unique<Ort::Session>(
        env_, cfg.onnx_path.c_str(), session_opts_);

    // ── Resolve input/output node names ───────────────────────────────────────
    Ort::AllocatorWithDefaultOptions alloc;

    const size_t num_in  = session_->GetInputCount();
    const size_t num_out = session_->GetOutputCount();

    input_names_owned_.reserve(num_in);
    output_names_owned_.reserve(num_out);

    for (size_t i = 0; i < num_in; ++i) {
        input_names_owned_.emplace_back(
            session_->GetInputNameAllocated(i, alloc).get());
    }
    for (size_t i = 0; i < num_out; ++i) {
        output_names_owned_.emplace_back(
            session_->GetOutputNameAllocated(i, alloc).get());
    }

    // Build raw pointer views (ORT API takes const char**)
    input_names_.clear();
    output_names_.clear();
    for (const auto& n : input_names_owned_)  input_names_.push_back(n.c_str());
    for (const auto& n : output_names_owned_) output_names_.push_back(n.c_str());

    // ── Determine spatial dimensions from model metadata ─────────────────────
    // Input shape is typically [-1, 3, H, W]; fall back to config values if dynamic.
    auto in_info  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto in_shape = in_info.GetShape();  // vector<int64_t>

    if (in_shape.size() == 4) {
        if (in_shape[2] > 0) input_h_ = static_cast<int>(in_shape[2]);
        if (in_shape[3] > 0) input_w_ = static_cast<int>(in_shape[3]);
    }
    // Override with explicit config values when provided
    if (cfg.input_shape.height > 0) input_h_ = cfg.input_shape.height;
    if (cfg.input_shape.width  > 0) input_w_ = cfg.input_shape.width;

    // ── Pre-allocate staging buffer ───────────────────────────────────────────
    input_staging_.resize(
        static_cast<size_t>(max_batch_size_) * 3 * input_h_ * input_w_, 0.0f);

    loaded_ = true;

    LOG_INFO("OnnxBackend: loaded {} ({}×{}, batch={}, ep={})",
             cfg.onnx_path, input_h_, input_w_, max_batch_size_,
             device_ == DeviceType::MPS ? "CoreML/MPS" : "CPU");
}

void OnnxBackend::unloadModel() {
    session_.reset();
    input_names_owned_.clear();
    output_names_owned_.clear();
    input_names_.clear();
    output_names_.clear();
    input_staging_.clear();
    loaded_ = false;
}

void OnnxBackend::preprocess(const Batch& input, int batch_size) {
    for (int i = 0; i < batch_size; ++i) {
        const cv::Mat& src = input.frames[static_cast<size_t>(i)];

        // Resize → BGR→RGB → float [0,1]
        cv::Mat resized;
        cv::resize(src, resized, cv::Size(input_w_, input_h_));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat fp;
        rgb.convertTo(fp, CV_32F, 1.0 / 255.0);

        // HWC → CHW, write directly into staging buffer
        float* dst = input_staging_.data() +
                     static_cast<ptrdiff_t>(i) * 3 * input_h_ * input_w_;

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < input_h_; ++y) {
                const float* row = fp.ptr<float>(y);
                float* plane_row = dst + c * input_h_ * input_w_ + y * input_w_;
                for (int x = 0; x < input_w_; ++x) {
                    plane_row[x] = row[x * 3 + c];
                }
            }
        }
    }
}

void OnnxBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) {
        throw std::runtime_error("OnnxBackend::infer called without a loaded model");
    }

    const int bs = input.size();
    if (bs <= 0 || bs > max_batch_size_) {
        throw std::runtime_error(
            "OnnxBackend::infer: batch size " + std::to_string(bs) +
            " out of range [1, " + std::to_string(max_batch_size_) + "]");
    }

    preprocess(input, bs);

    // ── Build input tensor (CPU memory) ───────────────────────────────────────
    const std::array<int64_t, 4> in_shape{
        bs, 3, static_cast<int64_t>(input_h_), static_cast<int64_t>(input_w_)};
    const size_t in_elems =
        static_cast<size_t>(bs) * 3 * input_h_ * input_w_;

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_staging_.data(), in_elems,
        in_shape.data(), in_shape.size());

    // ── Run inference ─────────────────────────────────────────────────────────
    auto out_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &in_tensor, 1,
        output_names_.data(), output_names_.size());

    // ── Copy first output tensor into the flat output vector ──────────────────
    auto& out = out_tensors[0];
    const size_t total = out.GetTensorTypeAndShapeInfo().GetElementCount();
    output.resize(total);
    const float* src = out.GetTensorData<float>();
    std::copy(src, src + total, output.begin());
}

} // namespace infer
