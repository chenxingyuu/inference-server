#pragma once

#include <string>
#include <utility>
#include <vector>

namespace infer {

class ITaskManager {
public:
    enum class State { Stopped, Running };

    virtual ~ITaskManager() = default;
    virtual bool start(const std::string& task_id) = 0;
    virtual bool stop(const std::string& task_id) = 0;
    virtual std::vector<std::pair<std::string, State>> listTasks() const = 0;
};

} // namespace infer
