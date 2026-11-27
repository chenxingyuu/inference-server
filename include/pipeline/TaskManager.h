#pragma once

#include "common/Config.h"
#include "common/RuntimeState.h"
#include "pipeline/GraphExecutor.h"
#include "pipeline/ITaskManager.h"
#include "publisher/IPublisher.h"
#include "archive/FrameArchiver.h"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace infer {

class TaskManager : public ITaskManager {
public:
    using State = ITaskManager::State;
    using ModelProbe = std::function<void(const std::filesystem::path&)>;
    using GraphBuilder = std::function<std::unique_ptr<GraphExecutor>(const AppConfig&, const TaskConfig&)>;
    using ModelTest = std::function<nlohmann::json(const ModelConfig&, const std::string&)>;

    // config_path is stored for runtime-state persistence (sidecar JSON).
    // publishers is a non-owning registry: callers must ensure all pointed-to
    // publishers outlive the TaskManager.
    TaskManager(AppConfig cfg,
                std::string config_path,
                std::unordered_map<std::string, IPublisher*> publishers,
                std::shared_ptr<FrameArchiver> frame_archiver, ModelProbe model_probe = {},
                ModelTest model_test = {}, GraphBuilder graph_builder = {},
                std::chrono::milliseconds drain_timeout = std::chrono::seconds(30));

    ~TaskManager() override;
    void loadAll();
    void startAll();
    void stopAll();
    void startSourceMonitoring(SourceProbe probe,
                               std::chrono::milliseconds interval = std::chrono::seconds(30));
    void stopSourceMonitoring();
    bool start(const std::string& task_id) override;
    ActionResult requestStart(const std::string& task_id) override;
    bool stop(const std::string& task_id) override;

    std::vector<TaskRuntimeInfo>               listTasks()    const override;
    std::vector<SourceInfo>                    listSources()  const override;
    SourceProbeResponse probeSource(const std::string& id, const SourceProbe& probe) override;
    std::vector<PipelineInfo>                  listPipelines()const override;
    std::vector<ModelInfo>                     listModels()   const override;

    bool addSource(const PipelineSourceConfig& src)  override;
    bool removeSource(const std::string& id)         override;
    bool addPipeline(const PipelineConfig& pipeline)    override;
    bool removePipeline(const std::string& id)          override;
    bool updatePipeline(const PipelineConfig& pipeline) override;
    bool addTask(const TaskConfig& task, bool autostart = true) override;
    bool updateTask(const TaskConfig& task) override;
    ActionResult requestUpdate(const TaskConfig& task) override;
    bool removeTask(const std::string& id)           override;
    bool loadModel(const ModelConfig& model)         override;
    UnregisterResult unloadModel(const std::string& id) override;
    ModelDetail describeModel(const std::string& id) const override;
    nlohmann::json testModel(const std::string& id, const std::string& image) override;
    std::vector<RepositoryModelInfo> listRepositoryModels() const override;
    bool loadFromRepository(const std::string& id)   override;
    RepositoryModelExport describeModelExport(const std::string& id) const override;

    ModelPublishResult publishModel(const std::string& id, const std::string& token,
                                    const std::string& attempt) override;

private:
    // Each run owns its graph. It remains attached until its worker has been joined.
    struct Run {
        std::atomic<bool> cancelled{false};
        bool done{false};
        bool stop_requested{false};
        std::chrono::steady_clock::time_point startup_deadline;
        std::thread thread;
    };

    struct Entry {
        State state{State::Stopped};
        std::optional<TaskFailure> failure;
        std::shared_ptr<Run> run;
        bool removed{false};
    };

    bool validateTask(const TaskConfig& task) const;
    std::unique_ptr<GraphExecutor> buildGraph(const AppConfig& cfg, const TaskConfig& task);
    SourceInfo sourceInfoLocked(const PipelineSourceConfig& source) const;
    void trackSourceProbeLocked(const PipelineSourceConfig& source);
    SourceProbeResponse runSourceProbeLocked(const PipelineSourceConfig& source, const SourceProbe& probe,
                                             std::unique_lock<std::mutex>& lock);
    void watchSources(const SourceProbe& probe);
    void recordStartupTimeoutLocked(const std::string& task_id, Entry& entry);
    ActionResult startLocked(const std::string& task_id);
    void runTask(const TaskConfig& task, const AppConfig& cfg, const std::shared_ptr<Run>& run);
    void watchTasks();

    // Persist current runtime mutations to the sidecar JSON file.
    void persist();
    ModelReferences modelReferences(const std::string& id) const;
    bool referencesUnregisteredModel(const PipelineConfig& pipeline) const;

    AppConfig    cfg_;
    std::string  config_path_;
    RuntimeState runtime_state_;

    std::unordered_map<std::string, IPublisher*> publishers_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    ModelProbe model_probe_;
    ModelTest model_test_;
    GraphBuilder graph_builder_;
    std::chrono::milliseconds drain_timeout_;
    std::mutex model_check_mu_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> entries_;
    struct SourceProbeRecord {
        std::chrono::steady_clock::time_point next_check{std::chrono::steady_clock::now()};
        std::optional<SourceProbeResult> result;
    };
    std::unordered_map<std::string, std::shared_ptr<SourceProbeRecord>> source_probes_;
    std::unordered_set<std::string> active_source_probes_;
    std::condition_variable source_probe_cv_;
    std::chrono::milliseconds source_probe_interval_{std::chrono::seconds(30)};
    bool source_probe_exit_{false};
    std::vector<std::thread> source_probe_workers_;
    std::condition_variable lifecycle_cv_;
    bool stopping_all_{false};
    bool watcher_exit_{false};
    std::thread watcher_;
};

} // namespace infer
