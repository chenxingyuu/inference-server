#include "common/Config.h"
#include "common/Logger.h"
#include "stream/StreamPool.h"
#include "infer/BackendFactory.h"
#include "decoder/DecoderFactory.h"
#include "pipeline/BatchScheduler.h"
#include "pipeline/InferWorker.h"
#include "publisher/KafkaPublisher.h"

#include <csignal>
#include <atomic>
#include <memory>
#include <vector>
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

    // ── Create one InferWorker + BatchScheduler per model ─────────────────────
    struct ModelPipeline {
        std::unique_ptr<infer::InferWorker>    worker;
        std::unique_ptr<infer::BatchScheduler> scheduler;
    };

    std::vector<ModelPipeline> pipelines;
    pipelines.reserve(cfg.models.size());

    for (const auto& model_cfg : cfg.models) {
        try {
            auto backend = infer::createBackend(model_cfg);
            auto decoder = infer::createDecoder(model_cfg);

            auto worker = std::make_unique<infer::InferWorker>(
                model_cfg,
                std::move(backend),
                std::move(decoder),
                *publisher);

            infer::InferWorker* worker_ptr = worker.get();
            auto scheduler = std::make_unique<infer::BatchScheduler>(
                model_cfg, pool,
                [worker_ptr](infer::Batch b) {
                    worker_ptr->enqueue(std::move(b));
                });

            worker->start();
            scheduler->start();

            pipelines.push_back({std::move(worker), std::move(scheduler)});
            LOG_INFO("Pipeline started for model {}", model_cfg.id);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to start pipeline for model {}: {}",
                      model_cfg.id, e.what());
        }
    }

    if (pipelines.empty()) {
        LOG_CRITICAL("No pipelines started — exiting");
        return 1;
    }

    LOG_INFO("All pipelines running. Streams: {}  Models: {}",
             cfg.streams.size(), pipelines.size());

    // ── Main wait loop ────────────────────────────────────────────────────────
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────
    LOG_INFO("Shutting down…");

    for (auto& p : pipelines) {
        p.scheduler->stop();
        p.worker->stop();
    }
    pool.stopAll();
    publisher->flush();

    LOG_INFO("inference-server stopped");
    return 0;
}
