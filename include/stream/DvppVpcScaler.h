#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>
#include <cstdint>
#include <functional>

namespace infer {

// DVPP VPC hardware resize (YUV420SP → YUV420SP), e.g. 1080p → 640×640 stretch.
// Runs on the caller's thread; caller must have set the correct aclrtContext.
class DvppVpcScaler {
public:
    using CreateChannelFn = std::function<aclError(acldvppChannelDesc*)>;
    using DestroyChannelFn = std::function<aclError(acldvppChannelDesc*)>;
    using SetContextFn  = std::function<aclError(aclrtContext)>;
    using VpcResizeFn = std::function<aclError(
        acldvppChannelDesc*, acldvppPicDesc*, acldvppPicDesc*,
        acldvppResizeConfig*, aclrtStream)>;

    struct VpcApiStub {
        SetContextFn     setCurrentContext;
        CreateChannelFn createChannel;
        DestroyChannelFn destroyChannel;
        VpcResizeFn vpcResize;
    };

    DvppVpcScaler() = default;
    ~DvppVpcScaler();

    DvppVpcScaler(const DvppVpcScaler&) = delete;
    DvppVpcScaler& operator=(const DvppVpcScaler&) = delete;

    void setVpcApiForTest(VpcApiStub stubs) { vpc_api_ = std::move(stubs); }

    // Create DVPP VPC channel + stream. Caller must have set aclrtSetCurrentContext.
    bool init(int device_id, aclrtContext context);
    void shutdown();

    bool initialized() const { return channel_desc_ != nullptr; }

    // Synchronous resize: input/output pic_desc must be fully configured (NV12).
    bool resize(acldvppPicDesc* input, acldvppPicDesc* output);

    acldvppChannelDesc* channelDescForTest() const { return channel_desc_; }
    aclrtStream streamForTest() const { return stream_; }

private:
    int                   device_id_{0};
    acldvppChannelDesc*   channel_desc_{nullptr};
    acldvppResizeConfig*  resize_cfg_{nullptr};
    aclrtStream           stream_{nullptr};
    VpcApiStub            vpc_api_{};
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
