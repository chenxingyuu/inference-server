#pragma once

#include "common/Config.h"
#include "pipeline/GraphExecutor.h"
#include "pipeline/ITaskManager.h"
#include "publisher/IPublisher.h"
#include "archive/FrameArchiver.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace infer {

class TaskManager : public ITaskManager {
public:
    using State = ITaskManager::State;

    TaskManager(const AppConfig& cfg,
                IPublisher& publisher,
                std::shared_ptr<FrameArchiver> frame_archiver);

    void loadAll();
    void startAll();
    void stopAll();
    bool start(const std::string& task_id);
    bool stop(const std::string& task_id);

    std::vector<std::pair<std::string, State>> listTasks() const;

private:
    struct Entry {
        std::unique_ptr<GraphExecutor> executor;
        State state{State::Stopped};
    };

    const AppConfig& cfg_;
    IPublisher& publisher_;
    std::shared_ptr<FrameArchiver> frame_archiver_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace infer
