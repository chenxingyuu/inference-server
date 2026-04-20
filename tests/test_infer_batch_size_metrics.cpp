#include <gtest/gtest.h>
#include "metrics/Metrics.h"

using namespace infer;

// Metrics is a singleton with no reset() — use unique model_id per test.

TEST(InferBatchSizeMetrics, RecordsCountAndSum) {
    auto& m = Metrics::get();
    m.recordInferBatchSize("bs_m1", 8);
    m.recordInferBatchSize("bs_m1", 16);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_batch_size"), std::string::npos);
    EXPECT_NE(out.find("infer_batch_size_count{model_id=\"bs_m1\"} 2"), std::string::npos);
    EXPECT_NE(out.find("infer_batch_size_sum{model_id=\"bs_m1\"} 24.000"), std::string::npos);
}

TEST(InferBatchSizeMetrics, BucketPlacementBatch16) {
    auto& m = Metrics::get();
    // batch=16 should land in le=16 bucket (assuming bounds include 16)
    m.recordInferBatchSize("bs_m2", 16);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_batch_size_bucket{model_id=\"bs_m2\",le=\"16\"} 1"), std::string::npos);
    EXPECT_NE(out.find("infer_batch_size_bucket{model_id=\"bs_m2\",le=\"8\"} 0"), std::string::npos);
}

TEST(InferBatchSizeMetrics, BucketPlacementBatch1) {
    auto& m = Metrics::get();
    m.recordInferBatchSize("bs_m3", 1);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_batch_size_bucket{model_id=\"bs_m3\",le=\"1\"} 1"), std::string::npos);
}

TEST(InferBatchSizeMetrics, MultipleModelIdsAreIndependent) {
    auto& m = Metrics::get();
    m.recordInferBatchSize("bs_ma", 4);
    m.recordInferBatchSize("bs_mb", 8);

    const std::string out = m.serialize();
    EXPECT_NE(out.find("infer_batch_size_count{model_id=\"bs_ma\"} 1"), std::string::npos);
    EXPECT_NE(out.find("infer_batch_size_count{model_id=\"bs_mb\"} 1"), std::string::npos);
}
