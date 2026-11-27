#include "server/UnixSocketServer.h"
#include "common/Config.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"

#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace infer {

using json = nlohmann::json;

namespace {

constexpr size_t kMaxRequestBytes = 12 * 1024 * 1024;

std::string requiredTaskString(const json& request, const char* field) {
    const auto value = request.find(field);
    if (value == request.end() || !value->is_string() ||
        value->get_ref<const std::string&>().empty()) {
        throw std::runtime_error(std::string(field) + " must be a non-empty string");
    }
    return value->get<std::string>();
}

bool taskBoolean(const json& request, const char* field, bool fallback) {
    const auto value = request.find(field);
    if (value == request.end()) return fallback;
    if (!value->is_boolean()) {
        throw std::runtime_error(std::string(field) + " must be a boolean");
    }
    return value->get<bool>();
}

int taskInteger(const json& request, const char* field, int fallback, int minimum) {
    const auto value = request.find(field);
    if (value == request.end()) return fallback;
    if (value->is_number()) {
        const auto number = value->get<double>();
        if (number >= minimum && number <= std::numeric_limits<int>::max() &&
            std::trunc(number) == number) {
            return static_cast<int>(number);
        }
    }
    throw std::runtime_error(std::string(field) + " must be an integer between " +
                             std::to_string(minimum) + " and " +
                             std::to_string(std::numeric_limits<int>::max()));
}

TaskConfig parseTaskConfig(const json& request) {
    TaskConfig task;
    task.id = requiredTaskString(request, "id");
    task.source_id = requiredTaskString(request, "source_id");
    task.pipeline_id = requiredTaskString(request, "pipeline_id");
    task.sample_fps = taskInteger(request, "sample_fps", task.sample_fps, 1);
    const auto sampling_mode = request.find("sampling_mode");
    if (sampling_mode != request.end()) {
        if (!sampling_mode->is_string()) {
            throw std::runtime_error("sampling_mode must be a string");
        }
        task.sampling_mode = parseSamplingMode(sampling_mode->get_ref<const std::string&>());
    }
    task.use_hwdec = taskBoolean(request, "use_hwdec", task.use_hwdec);
    task.use_ascend_dvpp = taskBoolean(request, "use_ascend_dvpp", task.use_ascend_dvpp);
    task.ascend_device_id = taskInteger(request, "ascend_device_id", task.ascend_device_id, 0);
    return task;
}

json modelInfoJson(const ModelInfo& model) {
    return {{"id", model.id}, {"backend", model.backend}, {"version", model.version},
            {"input_shape", model.input_shape}, {"batch_size", model.batch_size},
            {"instance_count", model.instance_count}, {"model_type", model.model_type},
            {"num_classes", model.num_classes}, {"conf_thresh", model.conf_thresh},
            {"nms_thresh", model.nms_thresh}, {"device_id", model.device_id},
            {"preferred_batch_sizes", model.preferred_batch_sizes},
            {"max_queue_delay_us", model.max_queue_delay_us},
            {"class_names", model.class_names}};
}

json sourceDiagnosticsJson(const SourceInfo& source) {
    const auto& availability = source.availability;
    json status = {{"status", availability.status}, {"basis", availability.basis}};
    if (availability.observed_at_ms) status["observed_at_ms"] = *availability.observed_at_ms;
    if (!availability.reason.empty()) status["reason"] = availability.reason;
    json probe = nullptr;
    if (source.last_probe) {
        const auto& last = *source.last_probe;
        probe = {{"passed", last.passed}, {"checked_at_ms", last.checked_at_ms},
                 {"elapsed_ms", last.elapsed_ms}};
        if (!last.reason.empty()) probe["reason"] = last.reason;
    }
    return {{"id", source.id}, {"availability", std::move(status)}, {"last_probe", std::move(probe)}};
}

json sourceInfoJson(const SourceInfo& source) {
    auto result = sourceDiagnosticsJson(source);
    result["url"] = source.url;
    result["state"] = source.state;
    result["reconnect_count"] = source.reconnect_count;
    const auto& config = source.config;
    result["config"] = {{"reconnect_delay_ms", config.reconnect_delay_ms},
                        {"max_reconnect_delay_ms", config.max_reconnect_delay_ms},
                        {"degraded_threshold", config.degraded_threshold},
                        {"max_reconnect_attempts", config.max_reconnect_attempts},
                        {"open_timeout_ms", config.open_timeout_ms},
                        {"read_timeout_ms", config.read_timeout_ms}};
    return result;
}

bool sendAll(int fd, const std::string& response) {
#ifdef MSG_NOSIGNAL
    constexpr int flags = MSG_NOSIGNAL;
#else
    constexpr int flags = 0;
#endif
    size_t sent = 0;
    while (sent < response.size()) {
        const auto n = ::send(fd, response.data() + sent, response.size() - sent, flags);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

} // namespace

UnixSocketServer::UnixSocketServer(std::string socket_path, ITaskManager& task_manager,
                                   ITaskManager::SourceProbe source_probe)
    : path_(std::move(socket_path))
    , task_manager_(task_manager)
    , source_probe_(std::move(source_probe)) {}

UnixSocketServer::~UnixSocketServer() { stop(); }

void UnixSocketServer::start() {
    if (running_.exchange(true)) return;

    ::unlink(path_.c_str());

    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0)
        throw std::runtime_error(std::string("UnixSocketServer: socket() failed: ") + std::strerror(errno));

    ::fcntl(server_fd_, F_SETFD, FD_CLOEXEC);

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    ::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        throw std::runtime_error(std::string("UnixSocketServer: bind() failed on ") + path_ + ": " + std::strerror(errno));
    }

    // Restrict socket access to the owning user only.
    ::chmod(path_.c_str(), 0600);

    if (::listen(server_fd_, 8) < 0) {
        ::close(server_fd_);
        server_fd_ = -1;
        ::unlink(path_.c_str());
        throw std::runtime_error(std::string("UnixSocketServer: listen() failed: ") + std::strerror(errno));
    }

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

    // Join all client threads before releasing the object.
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lock(clients_mu_);
        to_join.swap(client_threads_);
    }
    for (auto& t : to_join)
        if (t.joinable()) t.join();

    ::unlink(path_.c_str());
}

void UnixSocketServer::acceptLoop() {
    while (running_) {
        int client_fd = ::accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) break;
        ::fcntl(client_fd, F_SETFD, FD_CLOEXEC);
#ifdef SO_NOSIGPIPE
        const int no_sigpipe = 1;
        ::setsockopt(client_fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif

        std::lock_guard<std::mutex> lock(clients_mu_);
        client_threads_.emplace_back([this, client_fd] {
            handleClient(client_fd);
            ::close(client_fd);
        });
    }
}

void UnixSocketServer::handleClient(int fd) {
    char buf[4096];
    std::string data;
    while (true) {
        const auto n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return;
        const auto previous_size = data.size();
        data.append(buf, static_cast<size_t>(n));
        const auto pos = data.find('\n', previous_size);
        if ((pos == std::string::npos ? data.size() : pos) > kMaxRequestBytes) {
            sendAll(fd, "{\"status\":\"error\",\"code\":\"request_too_large\"}\n");
            return;
        }
        if (pos == std::string::npos) continue;
        try {
            sendAll(fd, dispatch(data.substr(0, pos)) + '\n');
        } catch (const std::exception& e) {
            LOG_ERROR("UnixSocketServer: command failed: {}", e.what());
            sendAll(fd, "{\"status\":\"error\",\"code\":\"internal_error\","
                        "\"message\":\"engine command failed\"}\n");
        }
        return;
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
        for (const auto& info : task_manager_.listTasks()) {
            json task = {
                {"id",               info.config.id},
                {"state",            taskStateName(info.state)},
                {"source_id",        info.config.source_id},
                {"pipeline_id",      info.config.pipeline_id},
                {"sample_fps",       info.config.sample_fps},
                {"sampling_mode",    info.config.sampling_mode == SamplingMode::TimeBased ? "time_based" : "frame_count"},
                {"use_hwdec",        info.config.use_hwdec},
                {"use_ascend_dvpp",  info.config.use_ascend_dvpp},
                {"ascend_device_id", info.config.ascend_device_id},
            };
            if (info.failure) task["failure"] = {{"code", info.failure->code}, {"message", info.failure->message}};
            if (info.cleanup_pending) task["cleanup_pending"] = true;
            arr.push_back(std::move(task));
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "start_task") {
        const auto id = req.value("id", std::string{});
        const auto result = task_manager_.requestStart(id);
        if (result == ITaskManager::ActionResult::Busy)
            return json({{"status", "error"}, {"code", "task_busy"}, {"message", "task cleanup is in progress"}}).dump();
        if (result == ITaskManager::ActionResult::NotFound)
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "task not found: " + id}}).dump();
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
            arr.push_back(sourceInfoJson(s));
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "probe_source") {
        const auto result = task_manager_.probeSource(requiredTaskString(req, "id"), source_probe_);
        if (!result.source) return json({{"status", "error"}, {"code", result.error}}).dump();
        auto data = sourceDiagnosticsJson(*result.source);
        data["reused_runtime"] = result.reused_runtime;
        return json({{"status", "ok"}, {"data", std::move(data)}}).dump();
    }

    if (cmd == "list_pipelines") {
        json arr = json::array();
        for (const auto& p : task_manager_.listPipelines()) {
            json nodes = json::array();
            for (const auto& n : p.nodes)
                nodes.push_back({{"id", n.id}, {"type", n.type}, {"with", n.with}});
            json edges = json::array();
            for (const auto& e : p.edges)
                edges.push_back({{"from", e.from}, {"to", e.to},
                                  {"capacity", e.capacity}, {"drop_policy", e.drop_policy}});
            arr.push_back({{"id", p.id}, {"nodes", nodes}, {"edges", edges}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "list_models") {
        json arr = json::array();
        for (const auto& m : task_manager_.listModels()) {
            arr.push_back(modelInfoJson(m));
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "list_repository") {
        json arr = json::array();
        for (const auto& m : task_manager_.listRepositoryModels()) {
            arr.push_back({{"id", m.id}, {"backend", m.backend}, {"version", m.version},
                           {"model_type", m.model_type}, {"batch_size", m.batch_size},
                           {"num_classes", m.num_classes}, {"conf_thresh", m.conf_thresh},
                           {"nms_thresh", m.nms_thresh}, {"instance_count", m.instance_count},
                           {"loaded", m.loaded}});
        }
        return json({{"status", "ok"}, {"data", arr}}).dump();
    }

    if (cmd == "describe_model" || cmd == "test_model") {
        if (!req.contains("id") || !req["id"].is_string() || req["id"].get_ref<const std::string&>().empty()) {
            return json({{"status", "error"}, {"code", "invalid_model_id"}, {"message", "invalid model ID"}}).dump();
        }
        const auto& id = req["id"].get_ref<const std::string&>();
        if (cmd == "test_model") {
            if (!req.contains("image") || !req["image"].is_string()) {
                return json({{"status", "error"}, {"code", "invalid_image"}, {"message", "missing image"}}).dump();
            }
            return task_manager_.testModel(id, req["image"].get_ref<const std::string&>()).dump();
        }

        const auto detail = task_manager_.describeModel(id);
        const auto& repo = detail.repository;
        const bool repo_absent = repo.code == "no_repository" || repo.code == "model_not_found" ||
                                 repo.code == "invalid_model_id";
        const bool repo_error = !repo.code.empty() && !repo_absent;
        if (!detail.model) {
            return json({{"status", "error"},
                         {"code", repo_error ? "model_repository_unavailable" : "model_not_found"},
                         {"message", repo_error ? "could not read the model repository" : "model not found"}}).dump();
        }

        auto pipelines = json::array();
        for (const auto& [pipeline_id, nodes] : detail.references.pipelines) {
            auto tasks = json::array();
            const auto related = detail.tasks_by_pipeline.find(pipeline_id);
            if (related != detail.tasks_by_pipeline.end()) {
                for (const auto& [task_id, state] : related->second)
                    tasks.push_back({{"id", task_id}, {"state", taskStateName(state)}});
            }
            pipelines.push_back({{"id", pipeline_id}, {"nodes", nodes.direct_nodes},
                                 {"cascade_nodes", nodes.cascade_nodes}, {"tasks", std::move(tasks)}});
        }
        auto cascade = json::array();
        for (const auto& child : detail.cascade_children) {
            cascade.push_back({{"model_id", child.model_id}, {"attribute_key", child.attribute_key},
                               {"trigger_classes", child.trigger_classes}, {"crop_expand", child.crop_expand}});
        }
        json repository = nullptr;
        if (repo.code.empty()) repository = {{"version", repo.version}, {"weight_paths", repo.weight_paths}};
        return json({{"status", "ok"}, {"data", {
            {"model", modelInfoJson(*detail.model)}, {"registered", detail.registered},
            {"repository", repository}, {"repository_error", repo_error},
            {"pipelines", pipelines}, {"cascade_parents", detail.references.cascade_parents},
            {"cascade_children", cascade}
        }}}).dump();
    }

    if (cmd == "describe_model_export") {
        if (!req.contains("id") || !req["id"].is_string()) {
            return json({{"status", "error"}, {"code", "invalid_model_id"}, {"message", "invalid model ID"}}).dump();
        }
        const auto model = task_manager_.describeModelExport(req["id"].get<std::string>());
        if (!model.code.empty()) {
            return json({{"status", "error"}, {"code", model.code}, {"message", model.message}}).dump();
        }
        return json({{"status", "ok"}, {"data", {
            {"version", model.version}, {"config_yaml", model.config_yaml},
            {"overrides_yaml", model.overrides_yaml}, {"weight_paths", model.weight_paths}
        }}}).dump();
    }

    if (cmd == "publish_model") {
        for (const auto* field : {"id", "token", "attempt"}) {
            if (!req.contains(field) || !req[field].is_string()) {
                return json({{"status", "error"}, {"code", "invalid_upload"},
                             {"message", std::string("missing or invalid field: ") + field}}).dump();
            }
        }
        const auto id = req["id"].get<std::string>();
        const auto result = task_manager_.publishModel(id, req["token"].get<std::string>(),
                                                       req["attempt"].get<std::string>());
        if (!result.code.empty()) {
            return json({{"status", "error"}, {"code", result.code}, {"message", result.message}}).dump();
        }
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "load_from_repository") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        if (!task_manager_.loadFromRepository(id))
            return json({{"status", "error"}, {"message", "not found or already loaded: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
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
        src.open_timeout_ms       = req.value("open_timeout_ms",       src.open_timeout_ms);
        src.read_timeout_ms       = req.value("read_timeout_ms",       src.read_timeout_ms);
        if (!task_manager_.addSource(src))
            return json({{"status", "error"}, {"message", "already exists: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "update_source") {
        try {
            const auto id = requiredTaskString(req, "id");
            const auto setting = [&](const char* field, int minimum) -> std::optional<int> {
                if (!req.contains(field)) return std::nullopt;
                return taskInteger(req, field, minimum, minimum);
            };
            const SourceUpdate update{requiredTaskString(req, "url"),
                                      setting("reconnect_delay_ms", 0),
                                      setting("max_reconnect_delay_ms", 0),
                                      setting("degraded_threshold", 1),
                                      setting("max_reconnect_attempts", 0),
                                      setting("open_timeout_ms", 1),
                                      setting("read_timeout_ms", 1)};
            const auto result = task_manager_.updateSource(id, update);
            switch (result.status) {
                case SourceUpdateStatus::Updated:
                    return json({{"status", "ok"}, {"id", id},
                                 {"data", sourceInfoJson(*result.source)}}).dump();
                case SourceUpdateStatus::NotFound:
                    return json({{"status", "error"}, {"code", "not_found"},
                                 {"message", "source not found"}}).dump();
                case SourceUpdateStatus::Busy:
                    return json({{"status", "error"}, {"code", "in_use"},
                                 {"message", "stop related tasks and wait for cleanup before editing"}}).dump();
                case SourceUpdateStatus::InvalidArgument:
                    return json({{"status", "error"}, {"code", "invalid_argument"},
                                 {"message", "invalid source settings or source type change"}}).dump();
                case SourceUpdateStatus::PersistenceFailed:
                    return json({{"status", "error"}, {"code", "persistence_failed"},
                                 {"message", "could not save the source configuration"}}).dump();
            }
        } catch (const std::runtime_error& error) {
            return json({{"status", "error"}, {"code", "invalid_argument"},
                         {"message", error.what()}}).dump();
        }
    }

    if (cmd == "remove_source") {
        const auto id = req.value("id", std::string{});
        const auto srcs = task_manager_.listSources();
        if (std::none_of(srcs.begin(), srcs.end(), [&](const auto& s){ return s.id == id; }))
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        if (!task_manager_.removeSource(id))
            return json({{"status", "error"}, {"code", "in_use"}, {"message", "source in use by a task: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "add_pipeline" || cmd == "update_pipeline") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        PipelineConfig p;
        p.id = id;
        try {
            for (const auto& n : req.value("nodes", json::array())) {
                StageConfig s;
                s.id   = n.value("id",   std::string{});
                s.type = n.value("type", std::string{});
                if (n.contains("with") && n["with"].is_object()) {
                    for (const auto& [k, v] : n["with"].items()) {
                        if (!v.is_string()) {
                            throw std::runtime_error("pipeline node '" + s.id + "': with." +
                                                     k + " must be a string");
                        }
                        s.with[k] = v.get<std::string>();
                    }
                }
                if (s.type == "infer.engine") {
                    parseInferEngineThresholdOverrides(s);
                }
                p.nodes.push_back(std::move(s));
            }
            for (const auto& e : req.value("edges", json::array())) {
                EdgeConfig ec;
                ec.from        = e.value("from", std::string{});
                ec.to          = e.value("to",   std::string{});
                ec.capacity    = e.value("capacity", ec.capacity);
                ec.drop_policy = parseEdgeDropPolicy(e.value("drop_policy", std::string{"block"}));
                p.edges.push_back(std::move(ec));
            }
        } catch (const std::exception& error) {
            return json({{"status", "error"}, {"code", "invalid_pipeline"}, {"message", error.what()}}).dump();
        }
        if (cmd == "add_pipeline") {
            if (!task_manager_.addPipeline(p))
                return json({{"status", "error"}, {"code", "invalid_pipeline"},
                             {"message", "pipeline already exists or references a model that was unregistered: " + id}}).dump();
            return json({{"status", "ok"}, {"id", id}}).dump();
        }
        const auto pipes = task_manager_.listPipelines();
        const bool exists =
            std::any_of(pipes.begin(), pipes.end(), [&](const auto& pi) { return pi.id == id; });
        if (!exists)
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        if (!task_manager_.updatePipeline(p)) {
            for (const auto& tri : task_manager_.listTasks()) {
                if (tri.config.pipeline_id == id &&
                    (tri.state == TaskState::Running || tri.state == TaskState::Starting || tri.cleanup_pending))
                    return json({{"status", "error"},
                                  {"code", "in_use"},
                                  {"message", "pipeline in use by a running task: " + id}})
                        .dump();
            }
            return json({{"status", "error"},
                         {"code", "invalid_pipeline"},
                         {"message", "failed to apply pipeline update: " + id}})
                .dump();
        }
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "remove_pipeline") {
        const auto id = req.value("id", std::string{});
        const auto pipes = task_manager_.listPipelines();
        if (std::none_of(pipes.begin(), pipes.end(), [&](const auto& p){ return p.id == id; }))
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        if (!task_manager_.removePipeline(id))
            return json({{"status", "error"}, {"code", "in_use"}, {"message", "pipeline in use by a task: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "add_task" || cmd == "update_task") {
        const bool adding = cmd == "add_task";
        TaskConfig task;
        bool autostart = true;
        try {
            task = parseTaskConfig(req);
            if (adding) autostart = taskBoolean(req, "autostart", true);
        } catch (const std::runtime_error& error) {
            return json({{"status", "error"}, {"code", "invalid_argument"}, {"message", error.what()}}).dump();
        }

        if (adding) {
            if (!task_manager_.addTask(task, autostart))
                return json({{"status", "error"}, {"message", "failed to add task: " + task.id}}).dump();
        } else {
            const auto result = task_manager_.requestUpdate(task);
            if (result == ITaskManager::ActionResult::Busy)
                return json({{"status", "error"}, {"code", "task_busy"}, {"message", "stop the task and wait for cleanup before editing"}}).dump();
            if (result != ITaskManager::ActionResult::Accepted)
                return json({{"status", "error"}, {"message", "not found or invalid dependencies: " + task.id}}).dump();
        }
        return json({{"status", "ok"}, {"id", task.id}}).dump();
    }

    if (cmd == "remove_task") {
        const auto id = req.value("id", std::string{});
        const auto tasks = task_manager_.listTasks();
        if (std::none_of(tasks.begin(), tasks.end(), [&](const auto& info){ return info.config.id == id; }))
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        if (!task_manager_.removeTask(id))
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "load_model") {
        const auto id = req.value("id", std::string{});
        if (id.empty())
            return json({{"status", "error"}, {"message", "missing field: id"}}).dump();
        ModelConfig m;
        m.id            = id;
        m.backend       = parseDeviceType(req.value("backend",    std::string{"cpu"}));
        m.version       = parseYOLOVersion(req.value("version",   std::string{"yolov8"}));
        m.model_type    = (req.value("model_type", std::string{"detector"}) == "classifier")
                          ? ModelType::Classifier : ModelType::Detector;
        m.onnx_path     = req.value("onnx_path",   std::string{});
        m.engine_path   = req.value("engine_path", std::string{});
        m.batch_size    = req.value("batch_size",    m.batch_size);
        m.conf_thresh   = req.value("conf_thresh",   m.conf_thresh);
        m.nms_thresh    = req.value("nms_thresh",    m.nms_thresh);
        m.num_classes   = req.value("num_classes",   m.num_classes);
        m.instance_count = req.value("instance_count", m.instance_count);
        if (req.contains("input_shape") && req["input_shape"].is_object()) {
            m.input_shape.channels = req["input_shape"].value("c", m.input_shape.channels);
            m.input_shape.height   = req["input_shape"].value("h", m.input_shape.height);
            m.input_shape.width    = req["input_shape"].value("w", m.input_shape.width);
        }
        if (req.contains("om_paths") && req["om_paths"].is_object()) {
            for (const auto& [k, v] : req["om_paths"].items()) {
                try { m.om_paths[std::stoi(k)] = v.get<std::string>(); }
                catch (...) {}
            }
        }
        if (!task_manager_.loadModel(m))
            return json({{"status", "error"}, {"message", "already exists: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    if (cmd == "unload_model") {
        const auto id = req.value("id", std::string{});
        const auto result = task_manager_.unloadModel(id);
        if (result == ITaskManager::UnregisterResult::NotFound)
            return json({{"status", "error"}, {"code", "not_found"}, {"message", "not found: " + id}}).dump();
        if (result == ITaskManager::UnregisterResult::InUse)
            return json({{"status", "error"}, {"code", "in_use"},
                         {"message", "model is referenced by a pipeline or registered model: " + id}}).dump();
        return json({{"status", "ok"}, {"id", id}}).dump();
    }

    return json({{"status", "error"}, {"message", "unknown command: " + cmd}}).dump();
}

} // namespace infer
