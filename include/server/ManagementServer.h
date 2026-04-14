#pragma once

#include "stream/StreamPool.h"
#include <memory>
#include <thread>
#include <atomic>

// Forward-declare to avoid pulling httplib.h into all translation units
namespace httplib { class Server; }

namespace infer {

// Lightweight HTTP management server backed by cpp-httplib.
//
// Endpoints:
//   GET  /healthz         → 200 OK  (liveness probe)
//   GET  /metrics         → Prometheus text format
//   GET  /streams         → JSON list of active stream IDs
//   POST /streams         → Add stream  body: {"id","url","model_id"[,"sample_fps","use_hwdec"]}
//   DELETE /streams/{id}  → Remove stream
class ManagementServer {
public:
    ManagementServer(int port, StreamPool& pool);
    ~ManagementServer();

    // Starts the server in a background thread (non-blocking).
    void start();

    // Stops the server and joins the background thread.
    void stop();

private:
    void registerHandlers();

    int                              port_;
    StreamPool&                      pool_;
    std::unique_ptr<httplib::Server> srv_;
    std::thread                      thread_;
    std::atomic<bool>                running_{false};
};

} // namespace infer
