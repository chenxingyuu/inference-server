#include <gtest/gtest.h>
#include "publisher/ControlPublisher.h"
#include <nlohmann/json.hpp>

using namespace infer;
using json = nlohmann::json;

TEST(ControlPublisher, StreamDroppedHasRequiredFields) {
    std::string raw = ControlPublisher::serializeStreamEvent(
        ControlEventType::StreamDropped,
        "cam_01",
        StreamState::RECONNECTING,
        /*engine_ts=*/1744704000.0,
        /*attempt=*/1,
        /*max_attempts=*/5,
        /*reason=*/"av_read_frame error");
    auto j = json::parse(raw);
    EXPECT_EQ(j["msg_type"], "stream_event");
    EXPECT_EQ(j["event_type"], "stream_dropped");
    EXPECT_EQ(j["stream_id"], "cam_01");
    EXPECT_EQ(j["stream_state"], "RECONNECTING");
    EXPECT_EQ(j["attempt"], 1);
    EXPECT_EQ(j["max_attempts"], 5);
    EXPECT_EQ(j["reason"], "av_read_frame error");
    EXPECT_DOUBLE_EQ(j["engine_ts"].get<double>(), 1744704000.0);
}

TEST(ControlPublisher, StreamRecoveredHasRequiredFields) {
    auto j = json::parse(ControlPublisher::serializeStreamEvent(
        ControlEventType::StreamRecovered,
        "cam_01",
        StreamState::STREAMING,
        /*engine_ts=*/1.0,
        /*attempt=*/2,
        /*max_attempts=*/5,
        /*reason=*/""));
    EXPECT_EQ(j["event_type"], "stream_recovered");
    EXPECT_EQ(j["stream_state"], "STREAMING");
}

TEST(ControlPublisher, StreamFailedTerminalHasRequiredFields) {
    auto j = json::parse(ControlPublisher::serializeStreamEvent(
        ControlEventType::StreamFailedTerminal,
        "cam_01",
        StreamState::FAILED,
        /*engine_ts=*/1.0,
        /*attempt=*/5,
        /*max_attempts=*/5,
        /*reason=*/"max reconnect attempts reached"));
    EXPECT_EQ(j["event_type"], "stream_failed_terminal");
    EXPECT_EQ(j["stream_state"], "FAILED");
}

