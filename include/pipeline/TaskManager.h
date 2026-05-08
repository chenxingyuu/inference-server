#pragma once

#include "common/Config.h"
#include "common/RuntimeState.h"
#include "pipeline/GraphExecutor.h"
#include "pipeline/ITaskManager.h"
#include "publisher/IPublisher.h"
#include "archive/FrameArchiver.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace infer {

class TaskManager : public ITaskManager {
public:
    using State = ITaskManager::State;

    // config_path is stored for runtime-state persistence (sidecar JSON).
    TaskManager(AppConfig cfg,
                std::string config_path,
                IPublisher& publisher,
                std::shared_ptr<FrameArchiver> frame_archiver);

    void loadAll();
    void startAll();
    void stopAll();
    bool start(const std::string& task_id) override;
    bool stop(const std::string& task_id) override;

    std::vector<TaskRuntimeInfo>               listTasks()    const override;
    std::vector<SourceInfo>                    listSources()  const override;
    std::vector<PipelineInfo>                  listPipelines()const override;
    std::vector<ModelInfo>                     listModels()   const override;

    bool addSource(const PipelineSourceConfig& src)  override;
    bool removeSource(const std::string& id)         override;
    bool addPipeline(const PipelineConfig& pipeline)    override;
    bool removePipeline(const std::string& id)          override;
    bool updatePipeline(const PipelineConfig& pipeline) override;
    bool addTask(const TaskConfig& task, bool autostart = true) override;
    bool updateTask(const TaskConfig& task)          override;
    bool removeTask(const std::string& id)           override;
    bool loadModel(const ModelConfig& model)         override;
    bool unloadModel(const std::string& id)          override;
    std::vector<RepositoryModelInfo> listRepositoryModels() const override;
    bool loadFromRepository(const std::string& id)   override;

private:
    struct Entry {
        std::shared_ptr<GraphExecutor> executor;
        State state{State::Stopped};
    };

    // Build and optionally start a single task entry. Returns false if the
    // referenced source or pipeline doesn't exist in cfg_.
    bool buildEntry(const TaskConfig& task, bool autostart);

    // Persist current runtime mutations to the sidecar JSON file.
    void persist();

    AppConfig    cfg_;
    std::string  config_path_;
    RuntimeState runtime_state_;

    IPublisher& publisher_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace infer
