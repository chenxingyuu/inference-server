#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "pipeline/ResultMerger.h"
#include "publisher/IPublisher.h"
#include <thread>
#include <chrono>

using namespace infer;
using ::testing::_;

// ── Mock publisher ─────────────────────────────────────────────────────────

class MockPublisher : public IPublisher {
public:
    MOCK_METHOD(void, publish, (InferResult result), (override));
    MOCK_METHOD(void, flush,   (),                   (override));
};

// ── Helpers ────────────────────────────────────────────────────────────────

static InferResult makeResult(const std::string& stream_id, double frame_ts,
                               int num_detections = 0) {
    InferResult r;
    r.stream_id = stream_id;
    r.frame_ts  = frame_ts;
    r.frame_seq = 0;
    for (int i = 0; i < num_detections; ++i) {
        Detection d;
        d.class_id = i;
        r.detections.push_back(d);
    }
    return r;
}

// ── Tests ──────────────────────────────────────────────────────────────────

TEST(ResultMerger, ZeroExpectedPublishesImmediately) {
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_)).Times(1);

    ResultMerger merger(pub, 20);
    merger.registerPrimary(makeResult("cam_01", 1.0), 0);
}

TEST(ResultMerger, WaitsForExpectedAttributesThenPublishes) {
    MockPublisher pub;
    // publish must be called exactly once — only after addAttribute satisfies the count
    EXPECT_CALL(pub, publish(_)).Times(1);

    ResultMerger merger(pub, 100);
    merger.registerPrimary(makeResult("cam_01", 2.0, /*num_dets=*/1), 1);
    // publish not called yet (1 expected, 0 received)

    merger.addAttribute("cam_01", 2.0, 0, "vehicle_type", "sedan");
    // 1/1 received → publish triggered
}

TEST(ResultMerger, InjectsAttributeIntoCorrectDetection) {
    InferResult captured;
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_))
        .WillOnce(::testing::SaveArg<0>(&captured));

    ResultMerger merger(pub, 100);
    merger.registerPrimary(makeResult("cam_01", 3.0, /*num_dets=*/2), 2);

    merger.addAttribute("cam_01", 3.0, 0, "color", "red");
    merger.addAttribute("cam_01", 3.0, 1, "color", "blue");

    ASSERT_EQ(captured.detections.size(), 2u);
    EXPECT_EQ(captured.detections[0].attributes.at("color"), "red");
    EXPECT_EQ(captured.detections[1].attributes.at("color"), "blue");
}

TEST(ResultMerger, UnknownKeyInAddAttributeIsIgnored) {
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_)).Times(0);

    ResultMerger merger(pub, 100);
    // Nothing registered → addAttribute should just warn and not crash
    merger.addAttribute("cam_99", 9.0, 0, "x", "y");
}

TEST(ResultMerger, FlushExpiredPublishesPastDeadline) {
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_)).Times(1);

    ResultMerger merger(pub, /*max_wait_ms=*/1);  // 1ms deadline
    merger.registerPrimary(makeResult("cam_01", 4.0, 1), /*expected=*/1);

    // Wait for deadline to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    merger.flushExpired();
    // Should have published even though no addAttribute was called
}

TEST(ResultMerger, FlushBeforeDeadlineDoesNotPublish) {
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_)).Times(0);

    ResultMerger merger(pub, /*max_wait_ms=*/1000);
    merger.registerPrimary(makeResult("cam_01", 5.0, 1), /*expected=*/1);
    merger.flushExpired();  // deadline not yet expired
}

TEST(ResultMerger, MultipleStreamsConcurrent) {
    MockPublisher pub;
    EXPECT_CALL(pub, publish(_)).Times(2);

    ResultMerger merger(pub, 100);
    merger.registerPrimary(makeResult("cam_01", 6.0, 1), 1);
    merger.registerPrimary(makeResult("cam_02", 6.0, 1), 1);

    merger.addAttribute("cam_01", 6.0, 0, "type", "car");
    merger.addAttribute("cam_02", 6.0, 0, "type", "truck");
}

TEST(ResultMerger, ThreadSafetyRegisterAndAdd) {
    MockPublisher pub;
    // 10 primaries, each expects 1 attribute → 10 publishes
    EXPECT_CALL(pub, publish(_)).Times(10);

    ResultMerger merger(pub, 500);

    // Register all 10 primaries first
    for (int i = 0; i < 10; ++i) {
        merger.registerPrimary(makeResult("cam_01", static_cast<double>(i), 1), 1);
    }

    // Add attributes from 10 threads in parallel
    std::vector<std::thread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&merger, i] {
            merger.addAttribute("cam_01", static_cast<double>(i), 0, "k", "v");
        });
    }
    for (auto& t : threads) t.join();
}
