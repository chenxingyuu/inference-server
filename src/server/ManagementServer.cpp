#include "server/ManagementServer.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"

// cpp-httplib single-header HTTP server
#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include <httplib.h>

#include <nlohmann/json.hpp>

namespace infer {

using json = nlohmann::json;

ManagementServer::ManagementServer(int port, StreamPool& pool)
    : port_(port)
    , pool_(pool)
    , srv_(std::make_unique<httplib::Server>())
{
    registerHandlers();
}

ManagementServer::~ManagementServer() {
    stop();
}

void ManagementServer::registerHandlers() {
    // ── GET /healthz ─────────────────────────────────────────────────────────
    srv_->Get("/healthz", [](const httplib::Request& /*req*/,
                              httplib::Response& res) {
        res.set_content("OK\n", "text/plain");
        res.status = 200;
    });

    // ── GET /metrics ─────────────────────────────────────────────────────────
    srv_->Get("/metrics", [](const httplib::Request& /*req*/,
                              httplib::Response& res) {
        std::string body = Metrics::get().serialize();
        res.set_content(body, "text/plain; version=0.0.4; charset=utf-8");
        res.status = 200;
    });

    // ── GET /streams ─────────────────────────────────────────────────────────
    srv_->Get("/streams", [this](const httplib::Request& /*req*/,
                                  httplib::Response& res) {
        auto ids = pool_.activeStreams();
        json arr = json::array();
        for (const auto& id : ids) arr.push_back(id);
        res.set_content(arr.dump(), "application/json");
        res.status = 200;
    });

    // ── POST /streams ─────────────────────────────────────────────────────────
    // Body: {"id":"cam_003","url":"rtsp://...","model_id":"yolov8n_trt"[,"sample_fps":5,"use_hwdec":true]}
    srv_->Post("/streams", [this](const httplib::Request& req,
                                   httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            StreamConfig sc;
            sc.id       = body.at("id").get<std::string>();
            sc.url      = body.at("url").get<std::string>();
            sc.model_id = body.at("model_id").get<std::string>();
            sc.sample_fps        = body.value("sample_fps", 5);
            sc.reconnect_delay_ms = body.value("reconnect_delay_ms", 3000);
            sc.use_hwdec         = body.value("use_hwdec", false);

            pool_.addStream(sc);
            LOG_INFO("ManagementServer: added stream {}", sc.id);

            json resp = {{"status", "ok"}, {"id", sc.id}};
            res.set_content(resp.dump(), "application/json");
            res.status = 201;
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
            res.status = 400;
        }
    });

    // ── DELETE /streams/{id} ──────────────────────────────────────────────────
    srv_->Delete(R"(/streams/(.+))", [this](const httplib::Request& req,
                                             httplib::Response& res) {
        std::string id = req.matches[1];
        pool_.removeStream(id);
        LOG_INFO("ManagementServer: removed stream {}", id);
        json resp = {{"status", "ok"}, {"id", id}};
        res.set_content(resp.dump(), "application/json");
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
