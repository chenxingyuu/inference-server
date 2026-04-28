#ifdef BUILD_ASCEND_BACKEND

#include "stream/DVPPDecoder.h"
#include "stream/IStreamDecoder.h"
#include "common/Types.h"
#include "common/Config.h"
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>

using namespace infer;

// ── Existing tests ─────────────────────────────────────────────────────────

TEST(DVPPDecoder, ImplementsIStreamDecoder) {
    static_assert(std::is_base_of<IStreamDecoder, DVPPDecoder>::value,
                  "DVPPDecoder must implement IStreamDecoder");
    SUCCEED();
}

TEST(DVPPDecoder, StartSetsRunning) {
    DVPPDecoder decoder;
    EXPECT_FALSE(decoder.running());

    decoder.startForTest([](Frame) {});
    EXPECT_TRUE(decoder.running());

    decoder.stop();
    EXPECT_FALSE(decoder.running());
}

TEST(DVPPDecoder, StopClearsRunning) {
    DVPPDecoder decoder;
    decoder.startForTest([](Frame) {});
    decoder.stop();
    EXPECT_FALSE(decoder.running());
}

TEST(DVPPDecoder, InitCreatesChannel) {
    DVPPDecoder decoder;

    std::atomic<int> create_count{0};
    decoder.setDvppApiForTest({
        .createChannel = [&create_count](AclVdecChannelDesc** p) -> aclError {
            *p = reinterpret_cast<AclVdecChannelDesc*>(uintptr_t{1});
            create_count.fetch_add(1);
            return ACL_SUCCESS;
        },
        .destroyChannel = [](AclVdecChannelDesc*) -> aclError {
            return ACL_SUCCESS;
        },
        .vdecProcess = nullptr,
    });

    bool ok = decoder.initChannelForTest(/*device_id=*/0, /*w=*/640, /*h=*/640);
    EXPECT_TRUE(ok);
    EXPECT_EQ(create_count.load(), 1);

    decoder.destroyChannelForTest();
}

TEST(DVPPDecoder, DestroyChannelOnStop) {
    DVPPDecoder decoder;

    std::atomic<int> destroy_count{0};
    decoder.setDvppApiForTest({
        .createChannel = [](AclVdecChannelDesc** p) -> aclError {
            *p = reinterpret_cast<AclVdecChannelDesc*>(uintptr_t{1});
            return ACL_SUCCESS;
        },
        .destroyChannel = [&destroy_count](AclVdecChannelDesc*) -> aclError {
            destroy_count.fetch_add(1);
            return ACL_SUCCESS;
        },
        .vdecProcess = nullptr,
    });

    decoder.initChannelForTest(0, 640, 640);
    decoder.destroyChannelForTest();

    EXPECT_EQ(destroy_count.load(), 1);
}

TEST(DVPPDecoder, CallbackBuildsAscendFrame) {
    std::atomic<bool> frame_received{false};
    bool is_ascend_set = false;

    struct TestCtx {
        FrameCallback cb;
        int           device_id{0};
    } ctx;
    ctx.device_id = 0;
    ctx.cb = [&frame_received, &is_ascend_set](Frame f) {
        frame_received.store(true);
        is_ascend_set = f.is_ascend;
    };

    DVPPDecoder::onDecoded(nullptr, nullptr, &ctx);

    EXPECT_TRUE(frame_received.load());
    EXPECT_TRUE(is_ascend_set);
}

// ── New tests ───────────────────────────────────────────────────────────────

// initChannel must accept is_h265=true without failure (new parameter).
TEST(DVPPDecoder, InitChannelH265Succeeds) {
    DVPPDecoder decoder;
    decoder.setDvppApiForTest({
        .createChannel = [](AclVdecChannelDesc** p) -> aclError {
            *p = reinterpret_cast<AclVdecChannelDesc*>(uintptr_t{1});
            return ACL_SUCCESS;
        },
        .destroyChannel = [](AclVdecChannelDesc*) -> aclError { return ACL_SUCCESS; },
        .vdecProcess = nullptr,
    });
    bool ok = decoder.initChannelForTest(/*device_id=*/0, /*w=*/1280, /*h=*/720,
                                         /*is_h265=*/true);
    EXPECT_TRUE(ok);
    decoder.destroyChannelForTest();
}

// alignedYuvSize() must use DVPP alignment (width→16, height→2) for the buffer.
TEST(DVPPDecoder, AlignedYuvSizeAlreadyAligned) {
    // 640x480 is already 16/2 aligned
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(640, 480), 640u * 480u * 3u / 2u);
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(1280, 720), 1280u * 720u * 3u / 2u);
}

TEST(DVPPDecoder, AlignedYuvSizeWidthPaddedTo16) {
    // 854 → aligned to 864
    const uint32_t aw = 864u, h = 480u;
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(854, h), aw * h * 3u / 2u);
    // 1366 → aligned to 1376
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(1366, 768), 1376u * 768u * 3u / 2u);
}

TEST(DVPPDecoder, AlignedYuvSizeOddHeightPaddedTo2) {
    // Odd height must be rounded up to even
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(640, 479), 640u * 480u * 3u / 2u);
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(640, 1079), 640u * 1080u * 3u / 2u);
}

// alignedYuvSize must be strictly larger than the raw unaligned size for
// non-aligned resolutions (i.e. the implementation is not using raw dims).
TEST(DVPPDecoder, AlignedYuvSizeLargerThanRawForUnaligned) {
    const uint32_t raw = 854u * 480u * 3u / 2u;
    EXPECT_GT(DVPPDecoder::alignedYuvSize(854, 480), raw);
}

// Multiple channel IDs must be unique across DVPPDecoder instances.
TEST(DVPPDecoder, ChannelIdsAreUnique) {
    const int n = 4;
    std::vector<uint32_t> ids;
    for (int i = 0; i < n; ++i) {
        DVPPDecoder decoder;
        ids.push_back(decoder.channelIdForTest());
    }
    std::sort(ids.begin(), ids.end());
    auto it = std::unique(ids.begin(), ids.end());
    EXPECT_EQ(it, ids.end()) << "Duplicate channel IDs detected";
}

#else
int main() { return 0; }
#endif
