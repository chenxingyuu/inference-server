#ifdef BUILD_ASCEND_BACKEND

#include "stream/DvppVpcScaler.h"
#include "stream/DVPPDecoder.h"
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <atomic>

using namespace infer;

static DvppVpcScaler::VpcApiStub makeStubVpc() {
    return {
        .setCurrentContext = [](aclrtContext) { return ACL_SUCCESS; },
        .createChannel = [](acldvppChannelDesc*) { return ACL_SUCCESS; },
        .destroyChannel = [](acldvppChannelDesc*) { return ACL_SUCCESS; },
        .vpcResize = nullptr,
    };
}

TEST(DvppVpc, AlignedYuvSizeForVpcOutput) {
    EXPECT_EQ(DVPPDecoder::alignedYuvSize(640, 640), 640u * 640u * 3u / 2u);
    EXPECT_GT(DVPPDecoder::alignedYuvSize(1920, 1080), 1920u * 1080u * 3u / 2u);
}

TEST(DvppVpc, InitCreatesChannelWithStub) {
    std::atomic<int> create_calls{0};
    DvppVpcScaler scaler;
    auto stub = makeStubVpc();
    stub.createChannel = [&](acldvppChannelDesc* desc) {
        ++create_calls;
        EXPECT_NE(desc, nullptr);
        return ACL_SUCCESS;
    };
    scaler.setVpcApiForTest(std::move(stub));
    EXPECT_TRUE(scaler.init(0, reinterpret_cast<aclrtContext>(0x1)));
    EXPECT_EQ(create_calls.load(), 1);
    EXPECT_NE(scaler.channelDescForTest(), nullptr);
    scaler.shutdown();
    EXPECT_EQ(scaler.channelDescForTest(), nullptr);
}

TEST(DvppVpc, ResizeInvokesStub) {
    std::atomic<int> resize_calls{0};
    DvppVpcScaler scaler;
    auto stub = makeStubVpc();
    stub.vpcResize = [&](acldvppChannelDesc*, acldvppPicDesc* in, acldvppPicDesc* out,
                         acldvppResizeConfig*, aclrtStream) {
        ++resize_calls;
        EXPECT_NE(in, nullptr);
        EXPECT_NE(out, nullptr);
        return ACL_SUCCESS;
    };
    scaler.setVpcApiForTest(std::move(stub));
    ASSERT_TRUE(scaler.init(0, reinterpret_cast<aclrtContext>(0x1)));
    acldvppPicDesc* in  = reinterpret_cast<acldvppPicDesc*>(0x1001);
    acldvppPicDesc* out = reinterpret_cast<acldvppPicDesc*>(0x1002);
    EXPECT_TRUE(scaler.resize(in, out));
    EXPECT_EQ(resize_calls.load(), 1);
}

TEST(DvppVpc, ResizeFailsWhenStubFails) {
    DvppVpcScaler scaler;
    auto stub = makeStubVpc();
    stub.vpcResize = [](acldvppChannelDesc*, acldvppPicDesc*, acldvppPicDesc*,
                        acldvppResizeConfig*, aclrtStream) {
        return ACL_ERROR_FAILURE;
    };
    scaler.setVpcApiForTest(std::move(stub));
    ASSERT_TRUE(scaler.init(0, reinterpret_cast<aclrtContext>(0x1)));
    EXPECT_FALSE(scaler.resize(reinterpret_cast<acldvppPicDesc*>(0x1),
                               reinterpret_cast<acldvppPicDesc*>(0x2)));
}

#endif // BUILD_ASCEND_BACKEND
