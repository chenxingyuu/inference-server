#include "server/UnixSocketServer.h"
#include "common/Config.h"
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

    if (cmd == "list_sources") {
        json arr = json::array();
        for (const auto& s : task_manager_.listSources()) {
            arr.push_back({{"id", s.id}, {"url", s.url}, {"state", s.state}, {"reconnect_count", s.reconnect_count}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "list_pipelines") {
        json arr = json::array();
        for (const auto& p : task_manager_.listPipelines()) {
            arr.push_back({{"id", p.id}, {"nodes", p.nodes}, {"edge_count", p.edge_count}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "list_models") {
        json arr = json::array();
        for (const auto& m : task_manager_.listModels()) {
            arr.push_back({{"id", m.id}, {"backend", m.backend}, {"version", m.version},
                           {"input_shape", m.input_shape}, {"batch_size", m.batch_size},
                           {"instance_count", m.instance_count}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "add_source") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        PipelineSourceConfig src;
        src.id                    = id;
        src.url                   = req.value("url", std::string{});
        src.reconnect_delay_ms    = req.value("reconnect_delay_ms",    src.reconnect_delay_ms);
        src.max_reconnect_delay_ms= req.value("max_reconnect_delay_ms",src.max_reconnect_delay_ms);
        src.degraded_threshold    = req.value("degraded_threshold",    src.degraded_threshold);
        src.max_reconnect_attempts= req.value("max_reconnect_attempts",src.max_reconnect_attempts);
        if (!task_manager_.addSource(src))
            return json({{"status", "error"}, {"message", "already exists: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "remove_source") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.removeSource(id))
            return json({{"status", "error"}, {"message", "not found or in use: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "add_pipeline") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        PipelineConfig p;
        p.id = id;
        for (const auto& n : req.value("nodes", json::array())) {
            StageConfig s;
            s.id   = n.value("id",   std::string{});
            s.type = n.value("type", std::string{});
            if (n.contains("with") && n["with"].is_object())
                for (const auto& [k, v] : n["with"].items())
                    s.with[k] = v.get<std::string>();
            p.nodes.push_back(std::move(s));
        }
        for (const auto& e : req.value("edges", json::array())) {
            EdgeConfig ec;
            ec.from = e.value("from", std::string{});
            ec.to   = e.value("to",   std::string{});
            p.edges.push_back(std::move(ec));
        }
        if (!task_manager_.addPipeline(p))
            return json({{"status", "error"}, {"message", "already exists: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "remove_pipeline") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.removePipeline(id))
            return json({{"status", "error"}, {"message", "not found or in use: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "add_task") {
        const auto id          = req.value("id",          std::string{});
        const auto source_id   = req.value("source_id",   std::string{});
        const auto pipeline_id = req.value("pipeline_id", std::string{});
        if (id.empty() || source_id.empty() || pipeline_id.empty())
            return json({{"status", "error"}, {"message", "missing required fields: id, source_id, pipeline_id"}}).dump();
        TaskConfig t;
        t.id          = id;
        t.source_id   = source_id;
        t.pipeline_id = pipeline_id;
        t.sample_fps  = req.value("sample_fps", t.sample_fps);
        if (!task_manager_.addTask(t))
            return json({{"status", "error"}, {"message", "failed to add task: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "remove_task") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.removeTask(id))
            return json({{"status", "error"}, {"message", "not found: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "load_model") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        ModelConfig m;
        m.id         = id;
        m.backend    = parseDeviceType(req.value("backend", std::string{"cpu"}));
        m.onnx_path  = req.value("onnx_path",   std::string{});
        m.engine_path= req.value("engine_path", std::string{});
        m.batch_size = req.value("batch_size",  m.batch_size);
        if (req.contains("input_shape") && req["input_shape"].is_object()) {
            m.input_shape.channels = req["input_shape"].value("c", m.input_shape.channels);
            m.input_shape.height   = req["input_shape"].value("h", m.input_shape.height);
            m.input_shape.width    = req["input_shape"].value("w", m.input_shape.width);
        }
        if (!task_manager_.loadModel(m))
            return json({{"status", "error"}, {"message", "already exists: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "unload_model") {
        const auto id = req.value("id", std::string{});
        if (!task_manager_.unloadModel(id))
            return json({{"status", "error"}, {"message", "not found: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    return json({{"status", "error"}, {"message", "unknown command: " + cmd}}).dump();
}

} // namespace infer
