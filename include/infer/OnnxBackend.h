#pragma once

#include "infer/IInferBackend.h"
#include "common/Config.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace infer {

/**
 * OnnxBackend — ONNX Runtime inference backend.
 *
 * Supports two execution providers, selected via DeviceType:
 *   DeviceType::CPU — CPU EP (cross-platform, no GPU required)
 *   DeviceType::MPS — CoreML EP on macOS Apple Silicon (Metal/MPS)
 *
 * Preprocessing mirrors AscendBackend: BGR→RGB, bilinear resize,
 * normalize to [0,1], HWC→CHW layout.  Output is a flat float buffer
 * identical in layout to TRTBackend, so all existing IYOLODecoder
 * implementations work without modification.
 *
 * Build requirement: -DBUILD_ONNX_BACKEND=ON
 * MPS/CoreML EP:    -DBUILD_ONNX_BACKEND_COREML=ON (macOS only)
 */
class OnnxBackend : public IInferBackend {
public:
    explicit OnnxBackend(DeviceType device);
    ~OnnxBackend() override;

    // Non-copyable, non-movable (owns Ort::Session which is not copyable)
    OnnxBackend(const OnnxBackend&)            = delete;
    OnnxBackend& operator=(const OnnxBackend&) = delete;

    void loadModel(const ModelConfig& cfg) override;
    void infer(const Batch& input, std::vector<float>& output) override;
    void unloadModel() override;

    int        maxBatchSize() const override { return max_batch_size_; }
    DeviceType deviceType()   const override { return device_; }
    bool       isLoaded()     const override { return loaded_; }

private:
    // Preprocess batch frames into input_staging_ (CPU float, CHW layout)
    void preprocess(const Batch& input, int batch_size);

    DeviceType device_;
    int        max_batch_size_{1};
    int        input_h_{640};
    int        input_w_{640};
    int        num_classes_{80};
    bool       loaded_{false};

    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;

    // Node names (owned strings + raw pointer views for the ORT API)
    std::vector<std::string>   input_names_owned_;
    std::vector<std::string>   output_names_owned_;
    std::vector<const char*>   input_names_;
    std::vector<const char*>   output_names_;

    std::vector<float> input_staging_;   // [max_batch * 3 * H * W]
};

} // namespace infer
