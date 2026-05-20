#include "common/Config.h"
#include "common/Logger.h"
#include "stream/FFmpegDecoder.h"
#include "pipeline/stages/DrawAndStreamStage.h"
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

// Returns an owning map of id → publisher. Callers retain ownership and pass
// a non-owning registry (raw pointers) to TaskManager.
std::unordered_map<std::string, std::unique_ptr<infer::IPublisher>>
buildPublisherRegistry(const std::vector<infer::PublisherConfig>& cfgs) {
    std::unordered_map<std::string, std::unique_ptr<infer::IPublisher>> registry;
    for (const auto& pc : cfgs) {
        std::unique_ptr<infer::IPublisher> pub;
        if (pc.type == "kafka") {
            pub = std::make_unique<infer::KafkaPublisher>(pc.kafka);
        } else if (pc.type == "redis") {
#ifdef BUILD_REDIS_PUBLISHER
            pub = std::make_unique<infer::RedisPublisher>(pc.redis);
#else
            throw std::runtime_error(
                "publishers[" + pc.id + "]: type=redis but binary built without "
                "BUILD_REDIS_PUBLISHER (re-run cmake with -DBUILD_REDIS_PUBLISHER=ON)");
#endif
        } else if (pc.type == "grpc") {
#ifdef BUILD_GRPC_PUBLISHER
            pub = std::make_unique<infer::GrpcPublisher>(pc.grpc);
#else
            throw std::runtime_error(
                "publishers[" + pc.id + "]: type=grpc but binary built without "
                "BUILD_GRPC_PUBLISHER (re-run cmake with -DBUILD_GRPC_PUBLISHER=ON)");
#endif
        } else {
            throw std::runtime_error("publishers[" + pc.id + "]: unknown type '" + pc.type + "'");
        }
        LOG_INFO("Publisher registered: id={} type={}", pc.id, pc.type);
        registry.emplace(pc.id, std::move(pub));
    }
    return registry;
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
    const auto task_count       = cfg.tasks.size();

    // Re-apply log level from config now that it is parsed.
    infer::initLogger(cfg.server.log_level);
    logConfiguredPipelinesAndTasks(cfg);

    // ── FFmpeg log level: env var > YAML > default ────────────────────────────
    {
        const char* env = std::getenv("FFMPEG_LOG_LEVEL");
        infer::setFfmpegLogLevel(env ? std::string(env) : cfg.server.ffmpeg_log_level);
    }
    infer::setFfmpegDecodeThreads(cfg.server.ffmpeg_decode_threads);
    infer::setFfmpegEncodeThreads(cfg.server.ffmpeg_encode_threads);

    // ── Signal handling ───────────────────────────────────────────────────────
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGPIPE, SIG_IGN);

    // ── Build publisher registry ──────────────────────────────────────────────
    std::unordered_map<std::string, std::unique_ptr<infer::IPublisher>> owned_publishers;
    try {
        owned_publishers = buildPublisherRegistry(cfg.publishers);
    } catch (const std::exception& e) {
        LOG_CRITICAL("Publisher init failed: {}", e.what());
        return 1;
    }

    // Build non-owning registry to pass to TaskManager (owned_publishers outlives it).
    std::unordered_map<std::string, infer::IPublisher*> publisher_registry;
    for (auto& [id, pub] : owned_publishers)
        publisher_registry.emplace(id, pub.get());

    auto frame_archiver = std::make_shared<infer::FrameArchiver>(cfg.frame_archive);
    infer::TaskManager task_manager(
        std::move(cfg), config_path, std::move(publisher_registry), frame_archiver);
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

    for (auto& [id, pub] : owned_publishers) pub->flush();
    LOG_INFO("inference-server stopped");
    return 0;
}
