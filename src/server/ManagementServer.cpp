#include "server/ManagementServer.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace infer {

using json = nlohmann::json;

namespace {
std::string stateStr(PipelineManager::State s) {
    return s == PipelineManager::State::Running ? "running" : "stopped";
}
} // namespace

ManagementServer::ManagementServer(int port, PipelineManager& pipeline_manager)
    : port_(port)
    , pipeline_manager_(pipeline_manager)
    , srv_(std::make_unique<httplib::Server>()) {
    registerHandlers();
}

ManagementServer::~ManagementServer() { stop(); }

void ManagementServer::registerHandlers() {
    srv_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK\n", "text/plain");
        res.status = 200;
    });

    srv_->Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(Metrics::get().serialize(), "text/plain; version=0.0.4; charset=utf-8");
        res.status = 200;
    });

    srv_->Get("/pipelines", [this](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (const auto& [id, state] : pipeline_manager_.listPipelines()) {
            arr.push_back({{"id", id}, {"state", stateStr(state)}});
        }
        res.set_content(arr.dump(), "application/json");
        res.status = 200;
    });

    srv_->Post(R"(/pipelines/([^/]+)/start)", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        if (!pipeline_manager_.start(id)) {
            res.set_content(json({{"error", "pipeline not found: " + id}}).dump(), "application/json");
            res.status = 404;
            return;
        }
        res.set_content(json({{"status", "ok"}, {"id", id}}).dump(), "application/json");
        res.status = 200;
    });

    srv_->Post(R"(/pipelines/([^/]+)/stop)", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        if (!pipeline_manager_.stop(id)) {
            res.set_content(json({{"error", "pipeline not found: " + id}}).dump(), "application/json");
            res.status = 404;
            return;
        }
        res.set_content(json({{"status", "ok"}, {"id", id}}).dump(), "application/json");
        res.status = 200;
    });
}

void ManagementServer::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] {
        LOG_INFO("ManagementServer: listening on port {}", port_);
        srv_->listen("0.0.0.0", port_);
        LOG_INFO("ManagementServer: stopped");
    });
}

void ManagementServer::stop() {
    if (!running_.exchange(false)) return;
    srv_->stop();
    if (thread_.joinable()) thread_.join();
}

} // namespace infer
