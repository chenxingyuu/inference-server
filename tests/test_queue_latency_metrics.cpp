#include <gtest/gtest.h>
#include "metrics/Metrics.h"

using namespace infer;

TEST(QueueLatencyMetrics, RecordsCountAndSum) {
    auto& m = Metrics::get();
    m.recordInferQueueLatency("ql_m1", 50.0);
    m.recordInferQueueLatency("ql_m1", 150.0);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_queue_latency_ms"), std::string::npos);
    EXPECT_NE(out.find("infer_queue_latency_ms_count{model_id=\"ql_m1\"} 2"), std::string::npos);
    EXPECT_NE(out.find("infer_queue_latency_ms_sum{model_id=\"ql_m1\"} 200.000"), std::string::npos);
}

TEST(QueueLatencyMetrics, BucketPlacement) {
    auto& m = Metrics::get();
    // 80ms should land in le=100 bucket, not le=50
    m.recordInferQueueLatency("ql_m2", 80.0);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_queue_latency_ms_bucket{model_id=\"ql_m2\",le=\"100\"} 1"), std::string::npos);
    EXPECT_NE(out.find("infer_queue_latency_ms_bucket{model_id=\"ql_m2\",le=\"50\"} 0"), std::string::npos);
}

TEST(QueueLatencyMetrics, MultipleModelsAreIndependent) {
    auto& m = Metrics::get();
    m.recordInferQueueLatency("ql_ma", 10.0);
    m.recordInferQueueLatency("ql_mb", 200.0);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_queue_latency_ms_count{model_id=\"ql_ma\"} 1"), std::string::npos);
    EXPECT_NE(out.find("infer_queue_latency_ms_count{model_id=\"ql_mb\"} 1"), std::string::npos);
}
