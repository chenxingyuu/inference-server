#include <gtest/gtest.h>
#include "metrics/Metrics.h"

using namespace infer;

// Metrics is a singleton with no reset() — use unique stage_id per test
// to avoid cross-test contamination.

TEST(SinkFfplayMetrics, JitterHistogramRecordsCountAndSum) {
    auto& m = Metrics::get();
    m.recordSinkFfplayJitter("jitter_s1", 60.0);
    m.recordSinkFfplayJitter("jitter_s1", 70.0);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_ffplay_frame_interval_ms"), std::string::npos);
    EXPECT_NE(out.find("sink_ffplay_frame_interval_ms_count{stage_id=\"jitter_s1\"} 2"),
              std::string::npos);
    EXPECT_NE(out.find("sink_ffplay_frame_interval_ms_sum{stage_id=\"jitter_s1\"} 130.000"),
              std::string::npos);
}

TEST(SinkFfplayMetrics, JitterHistogramBucketPlacement) {
    auto& m = Metrics::get();
    // 66ms falls in the 50<x<=100 bucket
    m.recordSinkFfplayJitter("jitter_s2", 66.0);

    const std::string out = m.serialize();
    // bucket le=100 must be >= 1, le=50 must be 0
    EXPECT_NE(out.find("sink_ffplay_frame_interval_ms_bucket{stage_id=\"jitter_s2\",le=\"100\"} 1"),
              std::string::npos);
    EXPECT_NE(out.find("sink_ffplay_frame_interval_ms_bucket{stage_id=\"jitter_s2\",le=\"50\"} 0"),
              std::string::npos);
}

TEST(SinkFfplayMetrics, QueueDepthGauge) {
    auto& m = Metrics::get();
    m.setSinkFfplayQueueDepth("qdepth_s1", 7);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_ffplay_queue_depth{stage_id=\"qdepth_s1\"} 7"), std::string::npos);
}

TEST(SinkFfplayMetrics, WrittenCounter) {
    auto& m = Metrics::get();
    m.incSinkFfplayFramesWritten("written_s1");
    m.incSinkFfplayFramesWritten("written_s1");
    m.incSinkFfplayFramesWritten("written_s1");

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_ffplay_frames_written_total{stage_id=\"written_s1\"} 3"),
              std::string::npos);
}

TEST(SinkFfplayMetrics, DroppedCounter) {
    auto& m = Metrics::get();
    m.incSinkFfplayFramesDropped("dropped_s1");
    m.incSinkFfplayFramesDropped("dropped_s1");

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_ffplay_frames_dropped_total{stage_id=\"dropped_s1\"} 2"),
              std::string::npos);
}

TEST(SinkFfplayMetrics, MultipleStageIdsAreIndependent) {
    auto& m = Metrics::get();
    m.incSinkFfplayFramesWritten("isolate_A");
    m.incSinkFfplayFramesWritten("isolate_A");
    m.incSinkFfplayFramesWritten("isolate_B");

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_ffplay_frames_written_total{stage_id=\"isolate_A\"} 2"),
              std::string::npos);
    EXPECT_NE(out.find("sink_ffplay_frames_written_total{stage_id=\"isolate_B\"} 1"),
              std::string::npos);
}

TEST(SinkMetrics, PublishAndStreamLatencyHistogramsAreExposed) {
    auto& m = Metrics::get();
    m.recordSinkPublishLatency("sink_latency_s1", 120.0);
    m.recordSinkStreamLatency("sink_latency_s1", 180.0);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("sink_publish_latency_ms"), std::string::npos);
    EXPECT_NE(out.find("sink_publish_latency_ms_count{stream_id=\"sink_latency_s1\"} 1"),
              std::string::npos);
    EXPECT_NE(out.find("sink_stream_latency_ms"), std::string::npos);
    EXPECT_NE(out.find("sink_stream_latency_ms_count{stream_id=\"sink_latency_s1\"} 1"),
              std::string::npos);
}
