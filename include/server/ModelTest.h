#pragma once

#include "common/Config.h"
#include "infer/IInferBackend.h"

#include <filesystem>
#include <nlohmann/json.hpp>

namespace infer {

inline constexpr size_t kModelTestMaxImageBytes = 8 * 1024 * 1024;
inline constexpr int kModelTestMaxImageSide = 4096;
inline constexpr size_t kModelTestMaxResponseBytes = 4 * 1024 * 1024;

// Uses a loaded backend. The image must already be decoded to BGR.
nlohmann::json evaluateModelImage(const ModelConfig& model, IInferBackend& backend, const cv::Mat& image);

int runModelTest(const std::filesystem::path& request_dir);
nlohmann::json testModelInChild(const std::string& executable, const ModelConfig& model,
                               const std::string& encoded_image);

} // namespace infer
