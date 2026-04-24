#include <gtest/gtest.h>
#include "publisher/GrpcPublisher.h"
#include "common/Types.h"
#include "inference.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

using namespace infer;

static InferResult makeResult(const std::string& stream_id = "cam_01",
                              float confidence = 0.9f) {
    InferResult r;
    r.stream_id  = stream_id;
    r.frame_ts   = 1234567890.5;
    r.infer_ts   = 1234567891.2;
    r.latency_ms = 42.0;
    r.model_id   = "yolov8m";
    r.frame_seq  = 7;
    Detection d;
    d.class_id   = 0;
    d.class_name = "person";
    d.confidence = confidence;
    d.bbox       = {10, 20, 100, 200};
    d.track_id   = 42;
    r.detections.push_back(d);
    return r;
}

static int freePort() {
    static std::atomic<int> base{50100};
    return base.fetch_add(1);
}

class GrpcPublisherTest : public ::testing::Test {
protected:
    std::unique_ptr<GrpcPublisher> publisher;
    std::unique_ptr<inference::InferenceService::Stub> stub;
    int port{0};

    void SetUp() override {
        port = freePort();
        GrpcConfig cfg;
        cfg.enabled = true;
        cfg.port    = port;
        publisher   = std::make_unique<GrpcPublisher>(cfg);

        auto channel = grpc::CreateChannel(
            "localhost:" + std::to_string(port),
            grpc::InsecureChannelCredentials());
        stub = inference::InferenceService::NewStub(channel);
    }
};

TEST_F(GrpcPublisherTest, ServerStartsAndListens) {
    // If we got here, the server started. Verify with a basic call.
    grpc::ClientContext ctx;
    inference::SubscribeRequest req;
    auto reader = stub->Subscribe(&ctx, req);
    ctx.TryCancel();
    inference::DetectionFrame frame;
    reader->Read(&frame);
    // Just verify we can connect and cancel without error.
}

TEST_F(GrpcPublisherTest, SubscriberReceivesPublishedFrame) {
    std::atomic<bool> received{false};
    inference::DetectionFrame got;

    std::thread subscriber([&] {
        grpc::ClientContext ctx;
        inference::SubscribeRequest req;
        auto reader = stub->Subscribe(&ctx, req);
        if (reader->Read(&got)) received.store(true);
        ctx.TryCancel();
        reader->Finish();
    });

    // Wait for subscriber to connect.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    publisher->publish(makeResult());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    subscriber.join();
    EXPECT_TRUE(received.load());
    EXPECT_EQ(got.stream_id(), "cam_01");
    EXPECT_DOUBLE_EQ(got.frame_ts(), 1234567890.5);
    EXPECT_EQ(got.model_id(), "yolov8m");
    EXPECT_EQ(got.frame_seq(), 7u);
}

TEST_F(GrpcPublisherTest, FilteredSubscriberReceivesOnlyMatchingFrames) {
    std::atomic<int> cam01_count{0};
    grpc::ClientContext ctx;

    std::thread subscriber([&] {
        inference::SubscribeRequest req;
        req.add_stream_ids("cam_01");
        auto reader = stub->Subscribe(&ctx, req);
        inference::DetectionFrame frame;
        while (reader->Read(&frame)) {
            if (frame.stream_id() == "cam_01") ++cam01_count;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    publisher->publish(makeResult("cam_02"));  // should NOT arrive
    publisher->publish(makeResult("cam_01"));  // should arrive
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ctx.TryCancel();
    subscriber.join();
    EXPECT_EQ(cam01_count.load(), 1);
}

TEST_F(GrpcPublisherTest, ProtoFieldsMatchInferResult) {
    inference::DetectionFrame got;
    std::atomic<bool> received{false};

    std::thread subscriber([&] {
        grpc::ClientContext ctx;
        inference::SubscribeRequest req;
        auto reader = stub->Subscribe(&ctx, req);
        if (reader->Read(&got)) received.store(true);
        ctx.TryCancel();
        reader->Finish();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    publisher->publish(makeResult("cam_03", 0.95f));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    subscriber.join();

    ASSERT_TRUE(received.load());
    ASSERT_EQ(got.detections_size(), 1);
    const auto& d = got.detections(0);
    EXPECT_EQ(d.class_id(), 0);
    EXPECT_EQ(d.class_name(), "person");
    EXPECT_FLOAT_EQ(d.confidence(), 0.95f);
    EXPECT_EQ(d.track_id(), 42);
    EXPECT_FLOAT_EQ(d.bbox().x1(), 10.f);
    EXPECT_FLOAT_EQ(d.bbox().y2(), 200.f);
}

TEST_F(GrpcPublisherTest, MultipleSubscribersEachReceive) {
    std::atomic<int> count1{0}, count2{0};

    auto runSubscriber = [&](std::atomic<int>& cnt) {
        grpc::ClientContext ctx;
        inference::SubscribeRequest req;
        auto reader = stub->Subscribe(&ctx, req);
        inference::DetectionFrame frame;
        if (reader->Read(&frame)) ++cnt;
        ctx.TryCancel();
        reader->Finish();
    };

    std::thread t1([&] { runSubscriber(count1); });
    std::thread t2([&] { runSubscriber(count2); });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    publisher->publish(makeResult());
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    t1.join();
    t2.join();
    EXPECT_EQ(count1.load(), 1);
    EXPECT_EQ(count2.load(), 1);
}

TEST_F(GrpcPublisherTest, FlushIsNoOp) {
    // flush() must not throw or block.
    EXPECT_NO_THROW(publisher->flush());
}
