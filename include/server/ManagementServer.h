#pragma once

#include "stream/StreamPool.h"
#include "pipeline/ModelManager.h"
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

// Forward-declare to avoid pulling httplib.h into all translation units
namespace httplib { class Server; }

namespace infer {

// Lightweight HTTP management server backed by cpp-httplib.
//
// Endpoints:
//   GET  /healthz                  → 200 OK  (liveness probe — always OK if process is alive)
//   GET  /health                   → 200/207/503 readiness probe with stream health summary
//   GET  /metrics                  → Prometheus text format
//   GET  /streams                  → JSON list of active stream IDs
//   POST /streams                  → Add stream
//   DELETE /streams/{id}           → Remove stream
//   GET  /streams/{id}/health      → Per-stream health detail (Phase 10)
//
//   GET  /models                   → JSON list of loaded models + state
//   POST /models                   → Load new model  body: ModelConfig JSON
//   DELETE /models/{id}            → Unload model (drains in-flight batches)
//   PUT  /models/{id}              → Hot-swap model  body: ModelConfig JSON
//   GET  /models/{id}/stats        → Per-model inference statistics
class ManagementServer {
public:
    ManagementServer(int port, StreamPool& pool, ModelManager& model_manager);
    ~ManagementServer();

    // Starts the server in a background thread (non-blocking).
    void start();

    // Stops the server and joins the background thread.
    void stop();

private:
    void registerHandlers();

    int                              port_;
    StreamPool&                      pool_;
    ModelManager&                    model_manager_;
    std::unique_ptr<httplib::Server> srv_;
    std::thread                      thread_;
    std::atomic<bool>                running_{false};
    std::chrono::steady_clock::time_point start_time_{std::chrono::steady_clock::now()};
};

} // namespace infer
