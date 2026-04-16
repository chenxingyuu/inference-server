#pragma once

#include "pipeline/PipelineManager.h"
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
//   GET  /healthz                  → 200 OK
//   GET  /metrics                  → Prometheus text format
//   GET  /pipelines                → list pipeline state
//   POST /pipelines/{id}/start     → start pipeline
//   POST /pipelines/{id}/stop      → stop pipeline
class ManagementServer {
public:
    ManagementServer(int port, PipelineManager& pipeline_manager);
    ~ManagementServer();

    // Starts the server in a background thread (non-blocking).
    void start();

    // Stops the server and joins the background thread.
    void stop();

private:
    void registerHandlers();

    int                              port_;
    PipelineManager&                 pipeline_manager_;
    std::unique_ptr<httplib::Server> srv_;
    std::thread                      thread_;
    std::atomic<bool>                running_{false};
    std::chrono::steady_clock::time_point start_time_{std::chrono::steady_clock::now()};
};

} // namespace infer
