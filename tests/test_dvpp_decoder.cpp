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

// ── Tests ──────────────────────────────────────────────────────────────────

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
        .createChannel = [&create_count](acldvppChannelDesc** p) -> aclError {
            // Return a fake non-null descriptor so the code doesn't null-check fail
            *p = reinterpret_cast<acldvppChannelDesc*>(uintptr_t{1});
            create_count.fetch_add(1);
            return ACL_SUCCESS;
        },
        .destroyChannel = [](acldvppChannelDesc*) -> aclError {
            return ACL_SUCCESS;
        },
        .vdecProcess = nullptr,
    });

    bool ok = decoder.initChannelForTest(/*device_id=*/0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(create_count.load(), 1);

    // Cleanup
    decoder.destroyChannelForTest();
}

TEST(DVPPDecoder, DestroyChannelOnStop) {
    DVPPDecoder decoder;

    std::atomic<int> destroy_count{0};
    decoder.setDvppApiForTest({
        .createChannel = [](acldvppChannelDesc** p) -> aclError {
            *p = reinterpret_cast<acldvppChannelDesc*>(uintptr_t{1});
            return ACL_SUCCESS;
        },
        .destroyChannel = [&destroy_count](acldvppChannelDesc*) -> aclError {
            destroy_count.fetch_add(1);
            return ACL_SUCCESS;
        },
        .vdecProcess = nullptr,
    });

    decoder.initChannelForTest(0);
    decoder.destroyChannelForTest();

    EXPECT_EQ(destroy_count.load(), 1);
}

TEST(DVPPDecoder, CallbackBuildsAscendFrame) {
    // Directly invoke the static onDecoded callback and verify it
    // constructs a Frame with is_ascend=true and calls the user callback.

    std::atomic<bool> frame_received{false};
    bool is_ascend_set = false;

    // Build a CallbackCtx-like structure.
    // DVPPDecoder::onDecoded expects user_data to be a pointer to a
    // struct with a FrameCallback field — we use a lambda wrapper.
    struct TestCtx {
        FrameCallback cb;
        int           device_id{0};
    } ctx;
    ctx.device_id = 0;
    ctx.cb = [&frame_received, &is_ascend_set](Frame f) {
        frame_received.store(true);
        is_ascend_set = f.is_ascend;
    };

    // Pass nullptr for both desc args — the callback must handle gracefully
    // (builds an AscendBuffer with null yuv_device when no real DVPP desc).
    DVPPDecoder::onDecoded(nullptr, nullptr, &ctx);

    EXPECT_TRUE(frame_received.load());
    EXPECT_TRUE(is_ascend_set);
}

#else
int main() { return 0; }
#endif
