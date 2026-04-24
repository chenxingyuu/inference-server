#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "publisher/MultiPublisher.h"
#include "publisher/IPublisher.h"
#include "common/Types.h"
#include <vector>
#include <thread>
#include <atomic>
#include <stdexcept>

using namespace infer;
using ::testing::_;
using ::testing::Exactly;
using ::testing::Throw;

class MockPublisher : public IPublisher {
public:
    MOCK_METHOD(void, publish, (InferResult), (override));
    MOCK_METHOD(void, flush,   (), (override));
};

static InferResult makeResult(const std::string& stream_id = "cam_01") {
    InferResult r;
    r.stream_id = stream_id;
    r.frame_ts  = 1.0;
    return r;
}

TEST(MultiPublisher, EmptyListThrows) {
    EXPECT_THROW(MultiPublisher({}), std::invalid_argument);
}

TEST(MultiPublisher, SinglePublisherForwardsPublish) {
    auto mock = std::make_unique<MockPublisher>();
    auto* raw = mock.get();
    EXPECT_CALL(*raw, publish(_)).Times(Exactly(1));
    EXPECT_CALL(*raw, flush()).Times(0);

    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::move(mock));
    MultiPublisher mp(std::move(pubs));
    mp.publish(makeResult());
}

TEST(MultiPublisher, TwoPublishersEachReceivePublish) {
    auto m1 = std::make_unique<MockPublisher>();
    auto m2 = std::make_unique<MockPublisher>();
    auto* r1 = m1.get();
    auto* r2 = m2.get();
    EXPECT_CALL(*r1, publish(_)).Times(Exactly(1));
    EXPECT_CALL(*r2, publish(_)).Times(Exactly(1));

    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::move(m1));
    pubs.push_back(std::move(m2));
    MultiPublisher mp(std::move(pubs));
    mp.publish(makeResult());
}

TEST(MultiPublisher, FlushCallsAllChildren) {
    auto m1 = std::make_unique<MockPublisher>();
    auto m2 = std::make_unique<MockPublisher>();
    auto* r1 = m1.get();
    auto* r2 = m2.get();
    EXPECT_CALL(*r1, flush()).Times(Exactly(1));
    EXPECT_CALL(*r2, flush()).Times(Exactly(1));
    EXPECT_CALL(*r1, publish(_)).Times(0);
    EXPECT_CALL(*r2, publish(_)).Times(0);

    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::move(m1));
    pubs.push_back(std::move(m2));
    MultiPublisher mp(std::move(pubs));
    mp.flush();
}

// A publisher that always throws on publish().
class ThrowingPublisher : public IPublisher {
public:
    void publish(InferResult) override { throw std::runtime_error("injected failure"); }
    void flush() override {}
};

// Counts how many times publish() is called.
class CountingPublisher : public IPublisher {
public:
    std::atomic<int> count{0};
    void publish(InferResult) override { ++count; }
    void flush() override {}
};

TEST(MultiPublisher, PartialFailureDoesNotSkipRemaining) {
    auto thrower  = std::make_unique<ThrowingPublisher>();
    auto counter  = std::make_unique<CountingPublisher>();
    auto* counter_raw = counter.get();

    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::move(thrower));
    pubs.push_back(std::move(counter));
    MultiPublisher mp(std::move(pubs));
    mp.publish(makeResult());  // should not throw even though first child does

    EXPECT_EQ(counter_raw->count.load(), 1);
}

TEST(MultiPublisher, SizeReturnsChildCount) {
    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::make_unique<CountingPublisher>());
    pubs.push_back(std::make_unique<CountingPublisher>());
    pubs.push_back(std::make_unique<CountingPublisher>());
    MultiPublisher mp(std::move(pubs));
    EXPECT_EQ(mp.size(), 3u);
}

TEST(MultiPublisher, ConcurrentPublishIsSafe) {
    auto counter = std::make_unique<CountingPublisher>();
    auto* counter_raw = counter.get();

    std::vector<std::unique_ptr<IPublisher>> pubs;
    pubs.push_back(std::move(counter));
    MultiPublisher mp(std::move(pubs));

    constexpr int kThreads = 10;
    constexpr int kPerThread = 100;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&mp] {
            for (int j = 0; j < kPerThread; ++j) {
                mp.publish(makeResult());
            }
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_EQ(counter_raw->count.load(), kThreads * kPerThread);
}
