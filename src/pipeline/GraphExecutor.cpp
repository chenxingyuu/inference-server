#include "pipeline/GraphExecutor.h"

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

void GraphExecutor::start() {
    if (running_.exchange(true)) return;
    for (auto& entry : stages_) {
        const auto id = entry.first;
        auto& stage = entry.second;
        stage->start();
        worker_threads_.emplace_back([this, id] { runNodeWorker(id); });
    }
}

void GraphExecutor::stop() {
    if (!running_.exchange(false)) return;
    // Unblock all queue pop() calls so worker threads can observe running_=false.
    for (auto& [_, q] : edges_) q->stop();
    // Before joining node workers, tell stages the graph is draining. Otherwise sinks
    // that manage their own worker (ffplay/ffmpeg) can still reconnect while graph
    // threads are winding down, which delays process exit after SIGINT.
    for (auto& [_, stage] : stages_) {
        stage->onGraphExecutorDraining();
    }
    // Join first: ensures no worker is inside stage->process() before we release resources.
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
    // Safe to release stage resources now that no worker thread is running.
    for (auto& [_, stage] : stages_) stage->stop();
}

void GraphExecutor::runNodeWorker(const std::string& node_id) {
    auto stage_it = stages_.find(node_id);
    if (stage_it == stages_.end()) return;
    auto* stage = stage_it->second.get();
    while (running_.load()) {
        bool processed = false;
        if (stage->isSource()) {
            EventEnvelope tick;
            stage->process(tick, [this, node_id](const EventEnvelope& out) {
                emitToDownstream(node_id, out);
            });
            processed = true;
        }
        const auto in_it = incoming_edge_keys_.find(node_id);
        if (in_it != incoming_edge_keys_.end()) {
            for (const auto& key : in_it->second) {
                auto q = edges_.at(key);
                EventEnvelope event;
                if (q->pop(event, 20)) {
                    stage->process(event, [this, node_id](const EventEnvelope& out) {
                        emitToDownstream(node_id, out);
                    });
                    processed = true;
                }
            }
        }
        if (!processed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void GraphExecutor::emitToDownstream(const std::string& from, const EventEnvelope& event) {
    auto out_it = outgoing_.find(from);
    if (out_it == outgoing_.end()) return;
    for (const auto& to : out_it->second) {
        auto q = edges_.at(edgeKey(from, to));
        q->push(event);
    }
}

} // namespace infer
