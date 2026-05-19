#include "common/Config.h"
#include "common/Logger.h"
#include "stream/FFmpegDecoder.h"
#include "pipeline/TaskManager.h"
#include "publisher/KafkaPublisher.h"
#include "publisher/MultiPublisher.h"
#ifdef BUILD_REDIS_PUBLISHER
#include "publisher/RedisPublisher.h"
#endif
#ifdef BUILD_GRPC_PUBLISHER
#include "publisher/GrpcPublisher.h"
#endif
#include "server/UnixSocketServer.h"
#include "archive/FrameArchiver.h"
#include <curl/curl.h>

#include <csignal>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
std::atomic<bool> g_shutdown{false};
void signalHandler(int) { g_shutdown.store(true); }
struct CurlGlobalGuard {
    CurlGlobalGuard() = default;
    ~CurlGlobalGuard() { curl_global_cleanup(); }
};

std::unique_ptr<infer::IPublisher> buildPublisher(const infer::PublishersConfig& cfg) {
    std::vector<std::unique_ptr<infer::IPublisher>> pubs;
    if (cfg.kafka.enabled)
        pubs.push_back(std::make_unique<infer::KafkaPublisher>(cfg.kafka));
#ifdef BUILD_REDIS_PUBLISHER
    if (cfg.redis.enabled)
        pubs.push_back(std::make_unique<infer::RedisPublisher>(cfg.redis));
#endif
#ifdef BUILD_GRPC_PUBLISHER
    if (cfg.grpc.enabled)
        pubs.push_back(std::make_unique<infer::GrpcPublisher>(cfg.grpc));
#endif
    if (pubs.empty()) {
        std::ostringstream msg;
        msg << "No result publisher could be constructed: ";
        if (!cfg.kafka.enabled)
            msg << "publishers.kafka.enabled is false; ";
#if !defined(BUILD_REDIS_PUBLISHER)
        if (cfg.redis.enabled)
            msg << "publishers.redis.enabled but binary built without BUILD_REDIS_PUBLISHER; ";
#endif
#if !defined(BUILD_GRPC_PUBLISHER)
        if (cfg.grpc.enabled)
            msg << "publishers.grpc.enabled but binary built without BUILD_GRPC_PUBLISHER; ";
#endif
        msg << "set publishers.kafka.enabled: true, or re-run cmake with "
               "-DBUILD_REDIS_PUBLISHER=ON -DBUILD_GRPC_PUBLISHER=ON (see Makefile configure-cpu).";
        throw std::runtime_error(msg.str());
    }
    if (pubs.size() == 1)
        return std::move(pubs[0]);
    return std::make_unique<infer::MultiPublisher>(std::move(pubs));
}

void logConfiguredPipelinesAndTasks(const infer::AppConfig& cfg) {
    LOG_INFO("Configured pipelines: {}", cfg.pipelines.size());
    for (const auto& pipeline : cfg.pipelines) {
        LOG_INFO("  pipeline id={} nodes={} edges={}",
                 pipeline.id,
                 pipeline.nodes.size(),
                 pipeline.edges.size());
    }

    LOG_INFO("Configured tasks: {}", cfg.tasks.size());
    for (const auto& task : cfg.tasks) {
        LOG_INFO("  task id={} source={} pipeline={} sample_fps={} hwdec={}",
                 task.id,
                 task.source_id,
                 task.pipeline_id,
                 task.sample_fps,
                 task.use_hwdec ? "true" : "false");
    }
}
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

    // cfg will be moved into TaskManager. Snapshot fields needed afterwards.
    const auto mgmt_socket_path = cfg.server.socket_path;
    const auto kafka_cfg        = cfg.publishers.kafka;
    const auto task_count       = cfg.tasks.size();

    // Re-apply log level from config now that it is parsed.
    infer::initLogger(cfg.server.log_level);
    logConfiguredPipelinesAndTasks(cfg);

    // ── FFmpeg log level: env var > YAML > default ────────────────────────────
    {
        const char* env = std::getenv("FFMPEG_LOG_LEVEL");
        infer::setFfmpegLogLevel(env ? std::string(env) : cfg.server.ffmpeg_log_level);
    }

    // ── Signal handling ───────────────────────────────────────────────────────
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGPIPE, SIG_IGN);

    // ── Create publisher ──────────────────────────────────────────────────────
    std::unique_ptr<infer::IPublisher> publisher;
    try {
        publisher = buildPublisher(cfg.publishers);
    } catch (const std::exception& e) {
        LOG_CRITICAL("Publisher init failed: {}", e.what());
        return 1;
    }

    auto frame_archiver = std::make_shared<infer::FrameArchiver>(cfg.frame_archive);
    infer::TaskManager task_manager(std::move(cfg), config_path, *publisher, frame_archiver);
    task_manager.loadAll();
    task_manager.startAll();

    // ── Start management HTTP server ──────────────────────────────────────────
    infer::UnixSocketServer mgmt_server(mgmt_socket_path, task_manager);
    mgmt_server.start();

    LOG_INFO("All tasks running. TaskCount: {} SocketPath: {}",
             task_count, mgmt_socket_path);

    // ── Main wait loop ────────────────────────────────────────────────────────
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    LOG_INFO("Shutting down…");

    // Stop inference graphs first so RTSP/ffplay sinks don't race with shutdown.
    task_manager.stopAll();

    mgmt_server.stop();

    publisher->flush();
    LOG_INFO("inference-server stopped");
    return 0;
}
