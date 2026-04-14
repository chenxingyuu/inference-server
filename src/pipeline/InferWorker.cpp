#include "pipeline/InferWorker.h"
#include "common/Logger.h"
#include <chrono>

namespace infer {

namespace {
double nowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}
} // namespace

InferWorker::InferWorker(const ModelConfig&            model_cfg,
                          std::unique_ptr<IInferBackend> backend,
                          std::unique_ptr<IYOLODecoder>  decoder,
                          IPublisher&                    publisher)
    : model_cfg_(model_cfg)
    , backend_(std::move(backend))
    , decoder_(std::move(decoder))
    , publisher_(publisher)
{}

InferWorker::~InferWorker() {
    stop();
}

void InferWorker::start() {
    if (running_.load()) return;
    backend_->loadModel(model_cfg_);
    stop_flag_ = false;
    thread_    = std::thread(&InferWorker::workerLoop, this);
}

void InferWorker::stop() {
    stop_flag_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    backend_->unloadModel();
    running_.store(false);
}

void InferWorker::enqueue(Batch batch) {
    std::unique_lock lock(mutex_);
    if (queue_.size() >= kMaxQueueSize) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    queue_.push(std::move(batch));
    lock.unlock();
    cv_.notify_one();
}

void InferWorker::workerLoop() {
    running_.store(true);
    LOG_INFO("InferWorker[{}]: started", model_cfg_.id);

    const InferShape shape = model_cfg_.input_shape;

    while (!stop_flag_.load()) {
        Batch batch;
        {
            std::unique_lock lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(20),
                         [this] { return !queue_.empty() || stop_flag_.load(); });
            if (queue_.empty()) continue;
            batch = std::move(queue_.front());
            queue_.pop();
        }

        if (batch.empty()) continue;

        try {
            std::vector<float> output;
            backend_->infer(batch, output);

            auto per_image = decoder_->decode(
                output.data(), batch.size(), shape,
                model_cfg_.conf_thresh, model_cfg_.nms_thresh);

            double infer_ts = nowEpoch();
            for (int i = 0; i < batch.size(); ++i) {
                InferResult r;
                r.stream_id  = batch.metas[i].stream_id;
                r.frame_ts   = batch.metas[i].capture_ts;
                r.infer_ts   = infer_ts;
                r.latency_ms = (infer_ts - r.frame_ts) * 1000.0;
                r.model_id   = model_cfg_.id;
                r.detections = std::move(per_image[i]);

                // Resolve class names if configured
                for (auto& d : r.detections) {
                    if (d.class_id < static_cast<int>(model_cfg_.class_names.size())) {
                        d.class_name = model_cfg_.class_names[d.class_id];
                    }
                }
                publisher_.publish(std::move(r));
            }
            processed_batches_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            LOG_ERROR("InferWorker[{}]: infer failed: {}", model_cfg_.id, e.what());
        }
    }

    LOG_INFO("InferWorker[{}]: stopped", model_cfg_.id);
    running_.store(false);
}

} // namespace infer
