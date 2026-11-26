#pragma once

#include "common/Config.h"
#include "pipeline/EdgeQueue.h"
#include "pipeline/IStage.h"
#include "pipeline/TaskState.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>

namespace infer {

class GraphExecutor {
public:
    explicit GraphExecutor(PipelineConfig cfg);
    ~GraphExecutor();

    void addStage(std::unique_ptr<IStage> stage);
    void build();
    void start(const std::atomic<bool>* cancelled = nullptr);
    bool inputsReady() const { return ready_sources_.load() == source_count_; }
    bool inputsEnded() const { return source_count_ > 0 && ended_sources_.load() == source_count_; }
    bool drained() const { return finished_nodes_.load() == stages_.size(); }
    std::optional<TaskFailure> failure() const;
    void stop();
    std::string graphText() const;
    bool running() const { return running_.load(); }
    const std::string& pipelineId() const { return cfg_.id; }

private:
    void recordFailure(const char* code, const std::string& node, const char* detail);
    void runNodeWorker(const std::string& node_id);
    void emitToDownstream(const std::string& from, const EventEnvelope& event);

    PipelineConfig cfg_;
    std::unordered_map<std::string, std::unique_ptr<IStage>> stages_;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_;
    std::unordered_map<std::string, std::vector<std::string>> incoming_edge_keys_;
    std::unordered_map<std::string, std::shared_ptr<EdgeQueue>> edges_;
    std::vector<std::thread> worker_threads_;
    std::vector<IStage*> started_stages_;
    std::size_t source_count_{0};
    std::atomic<std::size_t> ready_sources_{0};
    std::atomic<std::size_t> ended_sources_{0};
    std::atomic<std::size_t> finished_nodes_{0};
    mutable std::mutex failure_mutex_;
    std::optional<TaskFailure> failure_;
    std::atomic<bool> running_{false};
};

} // namespace infer
