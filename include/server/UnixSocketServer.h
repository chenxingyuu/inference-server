#pragma once

#include "pipeline/ITaskManager.h"
#include <atomic>
#include <string>
#include <thread>

namespace infer {

// Management server over a Unix domain socket.
//
// Protocol: newline-delimited JSON, one request → one response per connection.
//
// Commands:
//   {"cmd":"health"}
//   {"cmd":"metrics"}
//   {"cmd":"list_tasks"}
//   {"cmd":"start_task","id":"<id>"}
//   {"cmd":"stop_task","id":"<id>"}
class UnixSocketServer {
public:
    UnixSocketServer(std::string socket_path, ITaskManager& task_manager);
    ~UnixSocketServer();

    void start();
    void stop();

private:
    void acceptLoop();
    void handleClient(int fd);
    std::string dispatch(const std::string& line);

    std::string   path_;
    ITaskManager& task_manager_;
    int           server_fd_{-1};
    std::atomic<bool> running_{false};
    std::thread   thread_;
};

} // namespace infer
