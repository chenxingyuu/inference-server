#include "publisher/GrpcPublisher.h"
#include "common/Logger.h"

#include <grpcpp/server_builder.h>
#include <thread>

namespace infer {

GrpcPublisher::GrpcPublisher(const GrpcConfig& cfg) : cfg_(cfg) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "0.0.0.0:" + std::to_string(cfg_.port),
        grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    server_ = builder.BuildAndStart();
    if (!server_) {
        throw std::runtime_error("GrpcPublisher: failed to start server on port " +
                                 std::to_string(cfg_.port));
    }
    LOG_INFO("GrpcPublisher: listening on port {}", cfg_.port);
}

GrpcPublisher::~GrpcPublisher() {
    if (server_) {
        stop_.store(true);
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(2);
        server_->Shutdown(deadline);
        server_->Wait();
    }
}

void GrpcPublisher::publish(InferResult result) {
    inference::DetectionFrame frame = toProto(result);
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& sub : subscribers_) {
        if (!sub->active.load()) continue;
        if (!sub->matches(result.stream_id)) continue;
        if (!sub->writer->Write(frame)) {
            sub->active.store(false);
            ++dropped_;
            LOG_DEBUG("GrpcPublisher: subscriber disconnected for stream {}", result.stream_id);
        } else {
            ++published_;
        }
    }
    // Remove inactive subscribers.
    subscribers_.erase(
        std::remove_if(subscribers_.begin(), subscribers_.end(),
                       [](const auto& s) { return !s->active.load(); }),
        subscribers_.end());
}

int GrpcPublisher::activeSubscribers() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(subscribers_.size());
}

grpc::Status GrpcPublisher::Subscribe(
    grpc::ServerContext*                          ctx,
    const inference::SubscribeRequest*            request,
    grpc::ServerWriter<inference::DetectionFrame>* writer) {

    auto sub = std::make_shared<Subscriber>();
    for (const auto& id : request->stream_ids()) {
        sub->stream_ids.push_back(id);
    }
    sub->writer = writer;
    sub->active.store(true);

    {
        std::lock_guard<std::mutex> lock(mu_);
        subscribers_.push_back(sub);
    }

    // Block until subscriber disconnects, client cancels, or server is shutting down.
    while (!ctx->IsCancelled() && sub->active.load() && !stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        subscribers_.erase(
            std::remove_if(subscribers_.begin(), subscribers_.end(),
                           [&](const auto& s) { return s.get() == sub.get(); }),
            subscribers_.end());
    }

    return grpc::Status::OK;
}

inference::DetectionFrame GrpcPublisher::toProto(const InferResult& r) {
    inference::DetectionFrame frame;
    frame.set_stream_id(r.stream_id);
    frame.set_frame_ts(r.frame_ts);
    frame.set_infer_ts(r.infer_ts);
    frame.set_latency_ms(r.latency_ms);
    frame.set_model_id(r.model_id);
    frame.set_frame_seq(r.frame_seq);

    for (const auto& d : r.detections) {
        auto* det = frame.add_detections();
        det->set_class_id(d.class_id);
        det->set_class_name(d.class_name);
        det->set_confidence(d.confidence);
        det->set_track_id(d.track_id.value_or(0));
        auto* bbox = det->mutable_bbox();
        bbox->set_x1(d.bbox.x1);
        bbox->set_y1(d.bbox.y1);
        bbox->set_x2(d.bbox.x2);
        bbox->set_y2(d.bbox.y2);
        for (const auto& [k, v] : d.attributes) {
            (*det->mutable_attributes())[k] = v;
        }
    }
    return frame;
}

} // namespace infer
