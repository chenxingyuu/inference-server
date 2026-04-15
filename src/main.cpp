#include "common/Config.h"
#include "common/Logger.h"
#include "stream/StreamPool.h"
#include "infer/BackendFactory.h"
#include "decoder/DecoderFactory.h"
#include "pipeline/ModelManager.h"
#include "publisher/KafkaPublisher.h"
#include "server/ManagementServer.h"

#include <csignal>
#include <atomic>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
std::atomic<bool> g_shutdown{false};
void signalHandler(int) { g_shutdown.store(true); }
} // namespace

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "/config/config.yaml";

    // ── Init logging ──────────────────────────────────────────────────────────
    infer::initLogger("info");
    LOG_INFO("inference-server starting, config={}", config_path);

    // ── Load config ───────────────────────────────────────────────────────────
    infer::AppConfig cfg;
    try {
        cfg = infer::loadConfig(config_path);
    } catch (const std::exception& e) {
        LOG_CRITICAL("Failed to load config: {}", e.what());
        return 1;
    }

    // ── Signal handling ───────────────────────────────────────────────────────
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // ── Create publisher ──────────────────────────────────────────────────────
    std::unique_ptr<infer::IPublisher> publisher;
    try {
        publisher = std::make_unique<infer::KafkaPublisher>(cfg.kafka);
    } catch (const std::exception& e) {
        LOG_CRITICAL("KafkaPublisher init failed: {}", e.what());
        return 1;
    }

    // ── Create stream pool ────────────────────────────────────────────────────
    infer::StreamPool pool(cfg.server.max_streams);
    try {
        pool.start(cfg);
    } catch (const std::exception& e) {
        LOG_CRITICAL("StreamPool start failed: {}", e.what());
        return 1;
    }

    // ── Create ModelManager (Phase 7b-2) ─────────────────────────────────────
    // ModelManager owns all model pipelines and supports runtime load/unload/swap.
    // Pass all model configs so ModelManager can locate secondary model configs
    // for cascade wiring without requiring them to be loaded independently.
    infer::ModelManager model_manager(
        pool, *publisher,
        [](const infer::ModelConfig& mc) { return infer::createBackend(mc); },
        [](const infer::ModelConfig& mc) { return infer::createDecoder(mc); },
        cfg.models);

    // Collect cascade secondary model IDs — these are created internally by
    // their primary's startPipeline() and must not be loaded as standalone models.
    std::set<std::string> cascade_secondary_ids;
    for (const auto& m : cfg.models) {
        for (const auto& cas : m.cascade) {
            cascade_secondary_ids.insert(cas.model_id);
        }
    }

    // Load all non-secondary models defined in config at startup
    bool any_loaded = false;
    for (const auto& model_cfg : cfg.models) {
        if (cascade_secondary_ids.count(model_cfg.id)) {
            LOG_INFO("Model {} is a cascade secondary — loaded by its primary", model_cfg.id);
            continue;
        }
        try {
            model_manager.load(model_cfg);
            any_loaded = true;
            LOG_INFO("Pipeline started for model {} ({} instance(s))",
                     model_cfg.id, model_cfg.instance_count);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to start pipeline for model {}: {}",
                      model_cfg.id, e.what());
        }
    }

    if (!any_loaded) {
        LOG_CRITICAL("No pipelines started — exiting");
        return 1;
    }

    // ── Start management HTTP server ──────────────────────────────────────────
    infer::ManagementServer mgmt_server(
        cfg.server.management_port, pool, model_manager);
    mgmt_server.start();

    LOG_INFO("All pipelines running. Streams: {}  Models: {}  ManagementPort: {}",
             cfg.streams.size(), cfg.models.size(), cfg.server.management_port);

    // ── Main wait loop ────────────────────────────────────────────────────────
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    LOG_INFO("Shutting down…");

    mgmt_server.stop();

    // Unload all models (drains queues before destroying workers)
    for (const auto& [id, _] : model_manager.listModels()) {
        model_manager.unload(id);
    }

    pool.stopAll();
    publisher->flush();

    LOG_INFO("inference-server stopped");
    return 0;
}
