#include <gtest/gtest.h>
#include "stream/StreamHealthRegistry.h"
#include <thread>
#include <chrono>

using namespace infer;

// Helper: get a fresh registry (clear between tests)
class StreamHealthTest : public ::testing::Test {
protected:
    void SetUp() override {
        StreamHealthRegistry::get().clear();
    }
};

// ── State transitions ──────────────────────────────────────────────────────

TEST_F(StreamHealthTest, InitialStateIsConnecting) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    auto h = StreamHealthRegistry::get().getHealth("cam_01");
    EXPECT_EQ(h.state, StreamState::CONNECTING);
    EXPECT_EQ(h.reconnect_count, 0u);
    EXPECT_EQ(h.consecutive_failures, 0u);
}

TEST_F(StreamHealthTest, OpenedTransitionsToStreaming) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::STREAMING);
}

TEST_F(StreamHealthTest, DroppedTransitionsToReconnecting) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::RECONNECTING);
}

TEST_F(StreamHealthTest, ReconnectSucceededResetsFailuresAndIncrements) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_01");
    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    StreamHealthRegistry::get().onReconnectSucceeded("cam_01");

    auto h = StreamHealthRegistry::get().getHealth("cam_01");
    EXPECT_EQ(h.state, StreamState::STREAMING);
    EXPECT_EQ(h.consecutive_failures, 0u);
    EXPECT_EQ(h.reconnect_count, 1u);
}

TEST_F(StreamHealthTest, DegradedAfterThresholdFailures) {
    StreamHealthRegistry::get().onStreamAdded("cam_01", 3);
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_01");

    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::RECONNECTING);

    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::RECONNECTING);

    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::DEGRADED);
}

TEST_F(StreamHealthTest, RecoveryFromDegradedToStreaming) {
    StreamHealthRegistry::get().onStreamAdded("cam_01", 2);
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_01");
    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    ASSERT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::DEGRADED);

    StreamHealthRegistry::get().onReconnectSucceeded("cam_01");
    auto h = StreamHealthRegistry::get().getHealth("cam_01");
    EXPECT_EQ(h.state, StreamState::STREAMING);
    EXPECT_EQ(h.consecutive_failures, 0u);
    EXPECT_EQ(h.reconnect_count, 1u);
}

TEST_F(StreamHealthTest, RemovedTransitionsToStopped) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamRemoved("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::STOPPED);
}

TEST_F(StreamHealthTest, UnknownStreamReturnsStoppedHealth) {
    auto h = StreamHealthRegistry::get().getHealth("nonexistent");
    EXPECT_EQ(h.state, StreamState::STOPPED);
}

TEST_F(StreamHealthTest, TerminalFailureTransitionsToFailedAndCannotRecover) {
    StreamHealthRegistry::get().onStreamAdded("cam_01", /*degraded_threshold=*/2, /*max_reconnect_attempts=*/2);
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_01");

    // Hit threshold.
    StreamHealthRegistry::get().onReconnectFailed("cam_01");
    StreamHealthRegistry::get().onReconnectFailed("cam_01");

    auto h = StreamHealthRegistry::get().getHealth("cam_01");
    EXPECT_EQ(h.state, StreamState::FAILED);

    // After terminal failure, a reconnect success should NOT flip the state back.
    StreamHealthRegistry::get().onReconnectSucceeded("cam_01");
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::FAILED);
}

// ── Frame tracking ─────────────────────────────────────────────────────────

TEST_F(StreamHealthTest, FrameDecodedUpdatesTimestampAndCounter) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");

    StreamHealthRegistry::get().onFrameDecoded("cam_01", 1000.5);
    StreamHealthRegistry::get().onFrameDecoded("cam_01", 1001.0);

    auto h = StreamHealthRegistry::get().getHealth("cam_01");
    EXPECT_DOUBLE_EQ(h.last_frame_ts, 1001.0);
    EXPECT_EQ(h.frames_since_last_hb, 2u);
}

TEST_F(StreamHealthTest, ResetHbCountersClearsFrameCount) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onFrameDecoded("cam_01", 1000.0);
    StreamHealthRegistry::get().onFrameDecoded("cam_01", 1001.0);

    StreamHealthRegistry::get().resetHbCounters();
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").frames_since_last_hb, 0u);
}

// ── Multi-stream ───────────────────────────────────────────────────────────

TEST_F(StreamHealthTest, MultipleStreamsAreIndependent) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamAdded("cam_02");
    StreamHealthRegistry::get().onStreamOpened("cam_01");
    StreamHealthRegistry::get().onStreamDropped("cam_02");

    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_01").state, StreamState::STREAMING);
    EXPECT_EQ(StreamHealthRegistry::get().getHealth("cam_02").state, StreamState::RECONNECTING);
}

TEST_F(StreamHealthTest, GetAllHealthReturnsAllStreams) {
    StreamHealthRegistry::get().onStreamAdded("cam_01");
    StreamHealthRegistry::get().onStreamAdded("cam_02");
    StreamHealthRegistry::get().onStreamAdded("cam_03");

    auto all = StreamHealthRegistry::get().getAllHealth();
    EXPECT_EQ(all.size(), 3u);
}

// ── streamStateStr ─────────────────────────────────────────────────────────

TEST(StreamStateStr, AllStatesHaveStrings) {
    EXPECT_STREQ(streamStateStr(StreamState::CONNECTING),   "CONNECTING");
    EXPECT_STREQ(streamStateStr(StreamState::STREAMING),    "STREAMING");
    EXPECT_STREQ(streamStateStr(StreamState::RECONNECTING), "RECONNECTING");
    EXPECT_STREQ(streamStateStr(StreamState::DEGRADED),     "DEGRADED");
    EXPECT_STREQ(streamStateStr(StreamState::STOPPED),      "STOPPED");
    EXPECT_STREQ(streamStateStr(StreamState::FAILED),       "FAILED");
}
