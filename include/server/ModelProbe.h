#pragma once

#include <filesystem>
#include <string>

namespace infer {

void probeModelInChild(const std::string& executable, const std::filesystem::path& model_dir);
int runModelProbe(const std::filesystem::path& model_dir);

} // namespace infer
