#include "server/ManagementServer.h"
#include "metrics/Metrics.h"
#include "common/Logger.h"
#include "common/Config.h"

// cpp-httplib single-header HTTP server
#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include <httplib.h>

#include <nlohmann/json.hpp>

namespace infer {

using json = nlohmann::json;

// Helper: convert ModelManager::State to string
static std::string stateStr(ModelManager::State s) {
    switch (s) {
    case ModelManager::State::UNLOADED:  return "unloaded";
    case ModelManager::State::LOADING:   return "loading";
    case ModelManager::State::READY:     return "ready";
    case ModelManager::State::DRAINING:  return "draining";
    case ModelManager::State::UNLOADING: return "unloading";
    }
    return "unknown";
}

// Helper: parse ModelConfig from JSON body (minimal fields required by the API)
static ModelConfig modelConfigFromJson(const json& body) {
    ModelConfig m;
    m.id          = body.at("id").get<std::string>();
    m.engine_path = body.value("engine_path", "");
    m.batch_size  = body.value("batch_size", 16);
    m.conf_thresh = body.value("conf_thresh", 0.4f);
    m.nms_thresh  = body.value("nms_thresh", 0.45f);
    m.device_id   = body.value("device_id", 0);
    m.num_classes = body.value("num_classes", 80);
    m.buffer_pool_size     = body.value("buffer_pool_size", 4);
    m.instance_count       = body.value("instance_count", 1);
    m.max_queue_delay_us   = body.value("max_queue_delay_us", 10000);

    if (body.contains("version"))
        m.version = parseYOLOVersion(body["version"].get<std::string>());
    if (body.contains("backend"))
        m.backend = parseDeviceType(body["backend"].get<std::string>());

    if (body.contains("input_size")) {
        m.input_shape.height = body["input_size"][0].get<int>();
        m.input_shape.width  = body["input_size"][1].get<int>();
        m.input_shape.channels = 3;
        m.input_shape.batch    = m.batch_size;
    }
    if (body.contains("preferred_batch_sizes")) {
        for (const auto& s : body["preferred_batch_sizes"])
            m.preferred_batch_sizes.push_back(s.get<int>());
    }
    if (body.contains("device_ids")) {
        for (const auto& d : body["device_ids"])
            m.device_ids.push_back(d.get<int>());
    }
    if (body.contains("om_paths")) {
        for (auto it = body["om_paths"].begin(); it != body["om_paths"].end(); ++it)
            m.om_paths[std::stoi(it.key())] = it.value().get<std::string>();
    }
    return m;
}

ManagementServer::ManagementServer(int port, StreamPool& pool,
                                   ModelManager& model_manager)
    : port_(port)
    , pool_(pool)
    , model_manager_(model_manager)
    , srv_(std::make_unique<httplib::Server>())
{
    registerHandlers();
}

ManagementServer::~ManagementServer() {
    stop();
}

void ManagementServer::registerHandlers() {
    // ── GET /healthz ─────────────────────────────────────────────────────────
    srv_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK\n", "text/plain");
        res.status = 200;
    });

    // ── GET /metrics ─────────────────────────────────────────────────────────
    srv_->Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        std::string body = Metrics::get().serialize();
        res.set_content(body, "text/plain; version=0.0.4; charset=utf-8");
        res.status = 200;
    });

    // ── GET /streams ─────────────────────────────────────────────────────────
    srv_->Get("/streams", [this](const httplib::Request&, httplib::Response& res) {
        auto ids = pool_.activeStreams();
        json arr = json::array();
        for (const auto& id : ids) arr.push_back(id);
        res.set_content(arr.dump(), "application/json");
        res.status = 200;
    });

    // ── POST /streams ─────────────────────────────────────────────────────────
    srv_->Post("/streams", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            StreamConfig sc;
            sc.id                = body.at("id").get<std::string>();
            sc.url               = body.at("url").get<std::string>();
            sc.model_id          = body.at("model_id").get<std::string>();
            sc.sample_fps        = body.value("sample_fps", 5);
            sc.reconnect_delay_ms = body.value("reconnect_delay_ms", 3000);
            sc.use_hwdec         = body.value("use_hwdec", false);
            sc.tracker           = parseTrackerType(body.value("tracker", std::string("none")));

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

    // ── GET /models ───────────────────────────────────────────────────────────
    srv_->Get("/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = model_manager_.listModels();
        json arr    = json::array();
        for (const auto& [id, state] : models) {
            arr.push_back({{"id", id}, {"state", stateStr(state)}});
        }
        res.set_content(arr.dump(), "application/json");
        res.status = 200;
    });

    // ── POST /models ──────────────────────────────────────────────────────────
    // Load a new model. Body: ModelConfig JSON.
    srv_->Post("/models", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            auto cfg  = modelConfigFromJson(body);
            model_manager_.load(cfg);
            LOG_INFO("ManagementServer: loaded model {}", cfg.id);
            json resp = {{"status", "ok"}, {"id", cfg.id}, {"state", "ready"}};
            res.set_content(resp.dump(), "application/json");
            res.status = 201;
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
            res.status = 400;
        }
    });

    // ── DELETE /models/{id} ───────────────────────────────────────────────────
    // Unload model (drains in-flight batches before destroying).
    srv_->Delete(R"(/models/([^/]+)$)", [this](const httplib::Request& req,
                                               httplib::Response& res) {
        std::string id = req.matches[1];
        try {
            model_manager_.unload(id);
            LOG_INFO("ManagementServer: unloaded model {}", id);
            json resp = {{"status", "ok"}, {"id", id}};
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
            res.status = 400;
        }
    });

    // ── PUT /models/{id} ──────────────────────────────────────────────────────
    // Hot-swap: atomically replace running model with new config.
    srv_->Put(R"(/models/([^/]+)$)", [this](const httplib::Request& req,
                                            httplib::Response& res) {
        std::string id = req.matches[1];
        try {
            auto body    = json::parse(req.body);
            auto new_cfg = modelConfigFromJson(body);
            // Allow body to omit id; fall back to URL path segment
            if (new_cfg.id.empty()) new_cfg.id = id;

            model_manager_.swap(id, new_cfg);
            LOG_INFO("ManagementServer: swapped model {}", id);
            json resp = {{"status", "ok"}, {"id", id}, {"state", "ready"}};
            res.set_content(resp.dump(), "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
            res.status = 400;
        }
    });

    // ── GET /models/{id}/stats ────────────────────────────────────────────────
    srv_->Get(R"(/models/([^/]+)/stats)", [this](const httplib::Request& req,
                                                  httplib::Response& res) {
        std::string id    = req.matches[1];
        const auto* entry = model_manager_.find(id);
        if (!entry) {
            json err = {{"error", "model not found: " + id}};
            res.set_content(err.dump(), "application/json");
            res.status = 404;
            return;
        }
        json stats = {
            {"id",               id},
            {"state",            stateStr(entry->state)},
            {"instance_count",   entry->cfg.instance_count},
            {"batch_size",       entry->cfg.batch_size},
            {"processed_batches", entry->group ? entry->group->processedBatches() : 0},
            {"dropped_batches",   entry->group ? entry->group->droppedBatches()   : 0}
        };
        res.set_content(stats.dump(), "application/json");
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
