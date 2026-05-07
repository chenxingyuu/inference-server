#include "server/UnixSocketServer.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"

#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <thread>

namespace infer {

using json = nlohmann::json;

UnixSocketServer::UnixSocketServer(std::string socket_path, ITaskManager& task_manager)
    : path_(std::move(socket_path))
    , task_manager_(task_manager) {}

UnixSocketServer::~UnixSocketServer() { stop(); }

void UnixSocketServer::start() {
    if (running_.exchange(true)) return;

    ::unlink(path_.c_str());

    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    ::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
    ::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::listen(server_fd_, 8);

    thread_ = std::thread([this] {
        LOG_INFO("UnixSocketServer: listening on {}", path_);
        acceptLoop();
        LOG_INFO("UnixSocketServer: stopped");
    });
}

void UnixSocketServer::stop() {
    if (!running_.exchange(false)) return;
    ::shutdown(server_fd_, SHUT_RDWR);
    ::close(server_fd_);
    server_fd_ = -1;
    if (thread_.joinable()) thread_.join();
    ::unlink(path_.c_str());
}

void UnixSocketServer::acceptLoop() {
    while (running_) {
        int client_fd = ::accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) break;
        std::thread([this, client_fd] {
            handleClient(client_fd);
            ::close(client_fd);
        }).detach();
    }
}

void UnixSocketServer::handleClient(int fd) {
    char buf[4096];
    std::string data;
    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        data.append(buf, static_cast<size_t>(n));

        auto pos = data.find('\n');
        if (pos == std::string::npos) continue;

        std::string line = data.substr(0, pos);
        data.erase(0, pos + 1);

        std::string response = dispatch(line) + '\n';
        ::send(fd, response.data(), response.size(), 0);
    }
}

std::string UnixSocketServer::dispatch(const std::string& line) {
    json req;
    try {
        req = json::parse(line);
    } catch (...) {
        return json({{"status", "error"}, {"message", "invalid json"}}).dump();
    }

    const auto cmd = req.value("cmd", std::string{});

    if (cmd == "health") {
        return json({{"status", "ok"}}).dump();
    }

    if (cmd == "metrics") {
        return json({{"status", "ok"}, {"data", Metrics::get().serialize()}}).dump();
    }

    if (cmd == "list_tasks") {
        json arr = json::array();
        for (const auto& [id, state] : task_manager_.listTasks()) {
            arr.push_back({{"id", id},
                           {"state", state == ITaskManager::State::Running ? "running" : "stopped"}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "start_task") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.start(id))
            return json({{"status", "error"}, {"message", "task not found: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "stop_task") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.stop(id))
            return json({{"status", "error"}, {"message", "task not found: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    return json({{"status", "error"}, {"message", "unknown command: " + cmd}}).dump();
}

} // namespace infer
