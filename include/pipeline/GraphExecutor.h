#pragma once

#include "common/Config.h"
#include "pipeline/EdgeQueue.h"
#include "pipeline/IStage.h"
#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infer {

class GraphExecutor {
public:
    explicit GraphExecutor(PipelineConfig cfg);
    ~GraphExecutor();

    void addStage(std::unique_ptr<IStage> stage);
    void build();
    void start();
    void stop();
    bool running() const { return running_.load(); }
    const std::string& pipelineId() const { return cfg_.id; }

private:
    void runNodeWorker(const std::string& node_id);
    void emitToDownstream(const std::string& from, const EventEnvelope& event);

    PipelineConfig cfg_;
    std::unordered_map<std::string, std::unique_ptr<IStage>> stages_;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_;
    std::unordered_map<std::string, std::vector<std::string>> incoming_edge_keys_;
    std::unordered_map<std::string, std::shared_ptr<EdgeQueue>> edges_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
};

} // namespace infer
