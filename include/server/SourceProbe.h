#pragma once

#include "pipeline/ITaskManager.h"

namespace infer {

int runSourceProbe();
SourceProbeCode probeSourceInChild(const std::string& executable, const PipelineSourceConfig& source);

} // namespace infer
