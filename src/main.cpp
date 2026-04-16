#include "common/Config.h"
#include "common/Logger.h"
#include "pipeline/PipelineManager.h"
#include "publisher/KafkaPublisher.h"
#include "publisher/HeartbeatPublisher.h"
#include "publisher/ControlEventBus.h"
#include "server/ManagementServer.h"
#include "archive/FrameArchiver.h"
#include <curl/curl.h>

#include <csignal>
#include <atomic>
#include <memory>
#include <stdexcept>

namespace {
std::atomic<bool> g_shutdown{false};
void signalHandler(int) { g_shutdown.store(true); }
struct CurlGlobalGuard {
    CurlGlobalGuard() = default;
    ~CurlGlobalGuard() { curl_global_cleanup(); }
};
} // namespace

int main(int argc, char* argv[]) {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
        LOG_CRITICAL("Failed to initialize libcurl globals");
        return 1;
    }
    CurlGlobalGuard curl_guard;
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

    auto frame_archiver = std::make_shared<infer::FrameArchiver>(cfg.frame_archive);
    infer::PipelineManager pipeline_manager(cfg, *publisher, frame_archiver);
    pipeline_manager.loadAll();
    pipeline_manager.startAll();

    // ── Start heartbeat publisher (Phase 10) ─────────────────────────────────
    std::unique_ptr<infer::HeartbeatPublisher> heartbeat;
    try {
        heartbeat = std::make_unique<infer::HeartbeatPublisher>(cfg.kafka);
        heartbeat->start();
    } catch (const std::exception& e) {
        LOG_WARN("HeartbeatPublisher init failed (non-fatal): {}", e.what());
        heartbeat.reset();
    }

    // ── Start control publisher (Phase 14) ───────────────────────────────────
    std::shared_ptr<infer::ControlPublisher> control;
    try {
        control = std::make_shared<infer::ControlPublisher>(cfg.kafka.brokers, cfg.kafka.control_topic);
        infer::ControlEventBus::get().setPublisher(control);
    } catch (const std::exception& e) {
        LOG_WARN("ControlPublisher init failed (non-fatal): {}", e.what());
        control.reset();
    }

    // ── Start management HTTP server ──────────────────────────────────────────
    infer::ManagementServer mgmt_server(cfg.server.management_port, pipeline_manager);
    mgmt_server.start();

    LOG_INFO("All pipelines running. PipelineCount: {} ManagementPort: {}",
             cfg.pipelines.size(), cfg.server.management_port);

    // ── Main wait loop ────────────────────────────────────────────────────────
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    LOG_INFO("Shutting down…");

    mgmt_server.stop();

    if (heartbeat) heartbeat->stop();
    infer::ControlEventBus::get().clearPublisher();
    control.reset();

    pipeline_manager.stopAll();
    publisher->flush();
    LOG_INFO("inference-server stopped");
    return 0;
}
