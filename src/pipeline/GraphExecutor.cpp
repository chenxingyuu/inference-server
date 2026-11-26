#include "pipeline/GraphExecutor.h"

#include "common/Logger.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace infer {

namespace {
std::string edgeKey(const std::string& from, const std::string& to) {
    return from + "->" + to;
}
} // namespace

GraphExecutor::GraphExecutor(PipelineConfig cfg) : cfg_(std::move(cfg)) {}

GraphExecutor::~GraphExecutor() {
    stop();
}

void GraphExecutor::addStage(std::unique_ptr<IStage> stage) {
    stages_[stage->id()] = std::move(stage);
}

void GraphExecutor::build() {
    for (const auto& n : cfg_.nodes) {
        if (!stages_.count(n.id)) {
            throw std::runtime_error("missing stage implementation: " + n.id);
        }
    }
    for (const auto& e : cfg_.edges) {
        outgoing_[e.from].push_back(e.to);
        const auto key = edgeKey(e.from, e.to);
        edges_[key] = std::make_shared<EdgeQueue>(e);
        incoming_edge_keys_[e.to].push_back(key);
    }
}

std::string GraphExecutor::graphText() const {
    std::ostringstream oss;
    oss << "pipeline=" << cfg_.id << '\n';
    oss << "  nodes(" << cfg_.nodes.size() << ")\n";
    for (const auto& node : cfg_.nodes) {
        oss << "    - " << node.id << " [" << node.type << "]\n";
    }
    oss << "  edges(" << cfg_.edges.size() << ")\n";
    for (const auto& edge : cfg_.edges) {
        oss << "    - " << edge.from << " -> " << edge.to
            << " (cap=" << edge.capacity << ")\n";
    }
    return oss.str();
}

void GraphExecutor::start(const std::atomic<bool>* cancelled) {
    if (running_.exchange(true)) return;
    ready_sources_.store(0);
    ended_sources_.store(0);
    finished_nodes_.store(0);
    {
        std::lock_guard lock(failure_mutex_);
        failure_.reset();
    }
    source_count_ = std::count_if(stages_.begin(), stages_.end(), [](const auto& entry) {
        return entry.second->isSource();
    });
    for (auto& [_, queue] : edges_) queue->reopen();
    for (auto& [id, stage] : stages_) {
        if (cancelled && cancelled->load()) return;
        try {
            // A stage that throws can still have resources that must be stopped.
            started_stages_.push_back(stage.get());
            stage->start();
        } catch (const std::exception& error) {
            LOG_ERROR("Task {} node {} startup exception: {}", cfg_.id, id, error.what());
            recordFailure("startup_failed", id, "Pipeline node failed to start");
            return;
        } catch (...) {
            recordFailure("startup_failed", id, "unknown exception");
            return;
        }
    }
    for (const auto& [id, stage] : stages_) {
        if (cancelled && cancelled->load()) return;
        worker_threads_.emplace_back([this, node_id = id] { runNodeWorker(node_id); });
    }
}

void GraphExecutor::stop() {
    if (!running_.exchange(false)) return;
    for (auto& [_, queue] : edges_) queue->stop();
    for (auto* stage : started_stages_) stage->onGraphExecutorDraining();
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) thread.join();
    }
    worker_threads_.clear();
    for (auto it = started_stages_.rbegin(); it != started_stages_.rend(); ++it) {
        try {
            (*it)->stop();
        } catch (const std::exception& error) {
            LOG_ERROR("Task {} node {} cleanup exception: {}", cfg_.id, (*it)->id(), error.what());
            recordFailure("pipeline_error", (*it)->id(), "Pipeline node cleanup failed");
        } catch (...) {
            recordFailure("pipeline_error", (*it)->id(), "unknown exception during cleanup");
        }
    }
    started_stages_.clear();
}

std::optional<TaskFailure> GraphExecutor::failure() const {
    std::lock_guard lock(failure_mutex_);
    return failure_;
}

void GraphExecutor::recordFailure(const char* code, const std::string& node, const char* detail) {
    LOG_ERROR("Task {} node {}: {}", cfg_.id, node, detail);
    std::lock_guard lock(failure_mutex_);
    if (!failure_) failure_ = TaskFailure{code, "Pipeline node " + node + ": " + detail};
}

void GraphExecutor::runNodeWorker(const std::string& node_id) {
    auto& stage = *stages_.at(node_id);
    const auto incoming = incoming_edge_keys_.find(node_id);
    bool first_frame = false;
    bool input_ended = false;
    bool finishing = false;
    // Async emitters may outlive this worker's stack.
    const IStage::EmitFn emit = [this, node_id](const EventEnvelope& event) {
        emitToDownstream(node_id, event);
    };
    try {
        while (running_.load()) {
            if (const char* error = stage.terminalError()) {
                const std::string_view code(error);
                const char* message = "Video input ended unexpectedly";
                if (code == "source_retries_exhausted") message = "Video source retries exhausted";
                if (code == "inference_unavailable") message = "Primary inference workers could not recover";
                recordFailure(error, node_id, message);
                return;
            }
            if (stage.isSource() && stage.inputEnded() && !input_ended) {
                input_ended = true;
                ++ended_sources_;
            }
            bool processed = false;
            if (!finishing && stage.isSource() && !stage.inputDrained()) {
                stage.process({}, [&](const EventEnvelope& event) {
                    processed = true;
                    if (event.frame && !first_frame) {
                        first_frame = true;
                        ++ready_sources_;
                    }
                    emit(event);
                });
            }
            bool upstream_drained = incoming != incoming_edge_keys_.end();
            if (!finishing && incoming != incoming_edge_keys_.end()) {
                for (const auto& key : incoming->second) {
                    const auto& queue = edges_.at(key);
                    if (queue->inputDrained()) continue;
                    upstream_drained = false;
                    EventEnvelope event;
                    if (queue->pop(event, 20) && running_.load()) {
                        event.ingress_edge = key;
                        event.ingress_edge_queue_size = queue->size();
                        stage.process(event, emit);
                        processed = true;
                    }
                }
            }
            const bool no_live_root = incoming == incoming_edge_keys_.end() && inputsEnded();
            if (!finishing && (upstream_drained || no_live_root) &&
                (!stage.isSource() || stage.inputDrained())) {
                if (stage.isSource() && !first_frame) {
                    recordFailure("source_error", node_id, "Video input ended before the first frame");
                    return;
                }
                stage.finishInput();
                finishing = true;
            }
            if (finishing && stage.outputDrained()) {
                const auto outgoing = outgoing_.find(node_id);
                if (outgoing != outgoing_.end()) {
                    for (const auto& target : outgoing->second) edges_.at(edgeKey(node_id, target))->closeInput();
                }
                ++finished_nodes_;
                return;
            }
            if (!processed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception& error) {
        LOG_ERROR("Task {} node {} exception: {}", cfg_.id, node_id, error.what());
        recordFailure("pipeline_error", node_id, "Pipeline node execution failed");
    } catch (...) {
        recordFailure("pipeline_error", node_id, "Unknown pipeline node exception");
    }
}

void GraphExecutor::emitToDownstream(const std::string& from, const EventEnvelope& event) {
    auto out_it = outgoing_.find(from);
    if (out_it == outgoing_.end()) return;
    const auto& targets = out_it->second;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        auto q = edges_.at(edgeKey(from, targets[i]));
        if (i + 1 == targets.size()) {
            EventEnvelope moved = event;
            q->push(std::move(moved));
        } else {
            q->push(event);
        }
    }
}

} // namespace infer
