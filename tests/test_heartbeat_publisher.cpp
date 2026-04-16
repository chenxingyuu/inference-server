#include <gtest/gtest.h>
#include "publisher/HeartbeatPublisher.h"
#include <nlohmann/json.hpp>

using namespace infer;
using json = nlohmann::json;

// ── serializeStreamHb ──────────────────────────────────────────────────────

TEST(HeartbeatPublisher, StreamHbHasRequiredFields) {
    StreamHealth h;
    h.state               = StreamState::STREAMING;
    h.reconnect_count     = 2;
    h.consecutive_failures = 0;
    h.frames_since_last_hb = 25;

    std::string raw = HeartbeatPublisher::serializeStreamHb("cam_01", h, 1744704000.123);
    auto j = json::parse(raw);

    EXPECT_EQ(j["msg_type"],             "heartbeat");
    EXPECT_EQ(j["stream_id"],            "cam_01");
    EXPECT_EQ(j["stream_state"],         "STREAMING");
    EXPECT_EQ(j["reconnect_count"],      2);
    EXPECT_EQ(j["consecutive_failures"], 0);
    EXPECT_EQ(j["frames_since_last_hb"], 25);
    EXPECT_EQ(j["engine_healthy"],       true);
    EXPECT_DOUBLE_EQ(j["engine_ts"].get<double>(), 1744704000.123);
}

TEST(HeartbeatPublisher, StreamHbReflectsReconnectingState) {
    StreamHealth h;
    h.state               = StreamState::RECONNECTING;
    h.consecutive_failures = 3;

    auto j = json::parse(
        HeartbeatPublisher::serializeStreamHb("cam_02", h, 0.0));
    EXPECT_EQ(j["stream_state"],         "RECONNECTING");
    EXPECT_EQ(j["consecutive_failures"], 3);
}

TEST(HeartbeatPublisher, StreamHbReflectsDegradedState) {
    StreamHealth h;
    h.state               = StreamState::DEGRADED;
    h.consecutive_failures = 5;

    auto j = json::parse(
        HeartbeatPublisher::serializeStreamHb("cam_03", h, 0.0));
    EXPECT_EQ(j["stream_state"], "DEGRADED");
}

// ── serializeEngineHb ──────────────────────────────────────────────────────

TEST(HeartbeatPublisher, EngineHbHasRequiredFields) {
    std::string raw = HeartbeatPublisher::serializeEngineHb(3, 1744704000.0);
    auto j = json::parse(raw);

    EXPECT_EQ(j["msg_type"],       "engine_heartbeat");
    EXPECT_TRUE(j["stream_id"].is_null());
    EXPECT_EQ(j["active_streams"], 3);
    EXPECT_EQ(j["engine_healthy"], true);
    EXPECT_DOUBLE_EQ(j["engine_ts"].get<double>(), 1744704000.0);
}

TEST(HeartbeatPublisher, EngineHbWithZeroStreams) {
    auto j = json::parse(HeartbeatPublisher::serializeEngineHb(0, 0.0));
    EXPECT_EQ(j["active_streams"], 0);
}

// ── JSON is valid ──────────────────────────────────────────────────────────

TEST(HeartbeatPublisher, OutputIsValidJson) {
    StreamHealth h;
    h.state = StreamState::STOPPED;
    EXPECT_NO_THROW({
        auto j1 = json::parse(HeartbeatPublisher::serializeStreamHb("s", h, 1.0));
        auto j2 = json::parse(HeartbeatPublisher::serializeEngineHb(1, 1.0));
        (void)j1; (void)j2;
    });
}
