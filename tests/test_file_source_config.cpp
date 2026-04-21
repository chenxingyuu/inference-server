#include <gtest/gtest.h>
#include "common/Config.h"

using infer::SourceType;
using infer::StreamConfig;

// ─── SourceType enum ─────────────────────────────────────────────────────────

TEST(FileSourceConfig, SourceTypeEnumValuesAreDistinct) {
    EXPECT_NE(SourceType::RTSP, SourceType::File);
}

// ─── StreamConfig defaults ───────────────────────────────────────────────────

TEST(FileSourceConfig, StreamConfigDefaultsToRTSP) {
    StreamConfig cfg;
    EXPECT_EQ(cfg.source_type, SourceType::RTSP);
}

TEST(FileSourceConfig, StreamConfigLoopFileDefaultsFalse) {
    StreamConfig cfg;
    EXPECT_FALSE(cfg.loop_file);
}

// ─── StreamConfig field assignment ───────────────────────────────────────────

TEST(FileSourceConfig, CanSetSourceTypeFile) {
    StreamConfig cfg;
    cfg.source_type = SourceType::File;
    EXPECT_EQ(cfg.source_type, SourceType::File);
}

TEST(FileSourceConfig, CanEnableLoopFile) {
    StreamConfig cfg;
    cfg.source_type = SourceType::File;
    cfg.loop_file   = true;
    EXPECT_TRUE(cfg.loop_file);
}

TEST(FileSourceConfig, RTSPSourceTypeNotAffectedByLoopField) {
    StreamConfig cfg;
    cfg.loop_file = true;
    // source_type stays RTSP regardless
    EXPECT_EQ(cfg.source_type, SourceType::RTSP);
}
