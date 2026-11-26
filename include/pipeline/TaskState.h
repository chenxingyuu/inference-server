#pragma once

#include <string>

namespace infer {

enum class TaskState { Stopped, Starting, Running, Failed };

inline const char* taskStateName(TaskState state) {
    switch (state) {
        case TaskState::Stopped: return "stopped";
        case TaskState::Starting: return "starting";
        case TaskState::Running: return "running";
        case TaskState::Failed: return "failed";
    }
    return "stopped";
}

struct TaskFailure {
    std::string code;
    std::string message;
};

} // namespace infer
