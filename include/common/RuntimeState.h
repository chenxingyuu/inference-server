#pragma once

#include "common/Config.h"

#include <string>
#include <vector>

namespace infer {

// In-memory delta applied on top of the base YAML config.
// Serialised to {config_path}.runtime.json and re-applied on every startup.
struct RuntimeState {
    std::vector<PipelineSourceConfig> added_sources;
    std::vector<std::string>          removed_source_ids;

    std::vector<PipelineConfig>       added_pipelines;
    std::vector<std::string>          removed_pipeline_ids;

    std::vector<TaskConfig>           added_tasks;
    std::vector<std::string>          removed_task_ids;

    std::vector<ModelConfig>          added_models;
    std::vector<std::string>          removed_model_ids;
};

// Returns "{config_path}.runtime.json"
std::string runtimeStatePath(const std::string& config_path);

// Loads from disk; returns empty RuntimeState if the file doesn't exist.
RuntimeState loadRuntimeState(const std::string& path);

// Writes atomically (temp file + rename).
void saveRuntimeState(const RuntimeState& state, const std::string& path);

// Applies the delta to cfg in-place:
//  - removes entries whose id is in removed_*_ids
//  - appends added_* (skips duplicates by id)
void applyRuntimeState(AppConfig& cfg, const RuntimeState& state);

} // namespace infer
