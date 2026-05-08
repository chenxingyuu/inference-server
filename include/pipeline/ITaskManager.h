#pragma once

#include "common/Config.h"

#include <string>
#include <utility>
#include <vector>

namespace infer {

struct SourceInfo {
    std::string id;
    std::string url;
    std::string state;       // streamStateStr() value
    int         reconnect_count{0};
};

struct PipelineNodeInfo {
    std::string                          id;
    std::string                          type;
    std::map<std::string, std::string>   with;
};

struct PipelineEdgeInfo {
    std::string from;
    std::string to;
    int         capacity{256};
    std::string drop_policy;  // "block" | "drop_oldest" | "drop_newest"
};

struct PipelineInfo {
    std::string                   id;
    std::vector<PipelineNodeInfo> nodes;
    std::vector<PipelineEdgeInfo> edges;
};

struct ModelInfo {
    std::string id;
    std::string backend;
    std::string version;
    std::string input_shape;  // "CxHxW"
    int         batch_size{1};
    int         instance_count{1};
};

struct RepositoryModelInfo {
    std::string id;
    std::string backend;       // "trt" | "ascend" | "cpu"
    std::string version;       // "v5" | "v8" | "v11" | "v26"
    std::string model_type;    // "detector" | "classifier"
    int         batch_size{1};
    int         num_classes{80};
    float       conf_thresh{0.4f};
    float       nms_thresh{0.45f};
    int         instance_count{1};
    bool        loaded{false};
};

class ITaskManager {
public:
    enum class State { Stopped, Running };

    struct TaskRuntimeInfo {
        TaskConfig config;
        State      state{State::Stopped};
    };

    virtual ~ITaskManager() = default;
    virtual bool start(const std::string& task_id) = 0;
    virtual bool stop(const std::string& task_id) = 0;
    virtual std::vector<TaskRuntimeInfo> listTasks()    const = 0;
    virtual std::vector<SourceInfo>                    listSources()  const = 0;
    virtual std::vector<PipelineInfo>                  listPipelines()const = 0;
    virtual std::vector<ModelInfo>                     listModels()   const = 0;

    // Mutations — all return false on conflict (duplicate id or id not found).
    virtual bool addSource(const PipelineSourceConfig& src) = 0;
    virtual bool removeSource(const std::string& id) = 0;

    virtual bool addPipeline(const PipelineConfig& pipeline) = 0;
    virtual bool removePipeline(const std::string& id) = 0;
    virtual bool updatePipeline(const PipelineConfig& pipeline) = 0;

    virtual bool addTask(const TaskConfig& task, bool autostart = true) = 0;
    virtual bool updateTask(const TaskConfig& task) = 0;
    virtual bool removeTask(const std::string& id) = 0;

    virtual bool loadModel(const ModelConfig& model) = 0;
    virtual bool unloadModel(const std::string& id) = 0;

    // Repository operations — require model_repository to be configured.
    // Returns empty vector if repository path is unset or unreadable.
    virtual std::vector<RepositoryModelInfo> listRepositoryModels() const = 0;
    // Scan repository, find `id`, and load it. Returns false if not found or already loaded.
    virtual bool loadFromRepository(const std::string& id) = 0;
};

} // namespace infer
