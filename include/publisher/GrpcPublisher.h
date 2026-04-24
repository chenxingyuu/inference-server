#pragma once

#include "publisher/IPublisher.h"
#include "common/Config.h"
#include "inference.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>

namespace infer {

// gRPC server-streaming publisher.
// Clients call Subscribe(SubscribeRequest{stream_ids:[...]}) and receive a
// continuous DetectionFrame stream.  Slow subscribers are dropped when their
// write fails or when the pending frame count exceeds kMaxPendingFrames.
class GrpcPublisher final : public IPublisher,
                            public inference::InferenceService::Service {
public:
    explicit GrpcPublisher(const GrpcConfig& cfg);
    ~GrpcPublisher() override;

    GrpcPublisher(const GrpcPublisher&)            = delete;
    GrpcPublisher& operator=(const GrpcPublisher&) = delete;

    void publish(InferResult result) override;
    void flush() override {}  // no buffering; gRPC writes are synchronous per subscriber

    // Exposed for tests: number of currently active subscribers.
    int activeSubscribers() const;

    // gRPC service implementation.
    grpc::Status Subscribe(
        grpc::ServerContext*                          ctx,
        const inference::SubscribeRequest*            request,
        grpc::ServerWriter<inference::DetectionFrame>* writer) override;

private:
    struct Subscriber {
        std::vector<std::string>                          stream_ids;
        grpc::ServerWriter<inference::DetectionFrame>*    writer;
        std::atomic<bool>                                 active{true};

        bool matches(const std::string& sid) const {
            if (stream_ids.empty()) return true;
            for (const auto& id : stream_ids)
                if (id == sid) return true;
            return false;
        }
    };

    static inference::DetectionFrame toProto(const InferResult& r);

    static constexpr int kMaxPendingFrames = 64;

    GrpcConfig                                    cfg_;
    std::unique_ptr<grpc::Server>                 server_;

    mutable std::mutex                            mu_;
    std::vector<std::shared_ptr<Subscriber>>      subscribers_;

    std::atomic<uint64_t>                         published_{0};
    std::atomic<uint64_t>                         dropped_{0};
};

} // namespace infer
