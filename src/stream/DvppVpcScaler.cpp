#ifdef BUILD_ASCEND_BACKEND

#include "stream/DvppVpcScaler.h"
#include "common/Logger.h"

namespace infer {

DvppVpcScaler::~DvppVpcScaler() { shutdown(); }

bool DvppVpcScaler::init(int device_id, aclrtContext context) {
    shutdown();
    device_id_ = device_id;

    if (!context) {
        LOG_ERROR("DvppVpcScaler: null aclrtContext");
        return false;
    }
    aclError err = ACL_SUCCESS;
    if (vpc_api_.setCurrentContext) {
        err = vpc_api_.setCurrentContext(context);
    } else {
        err = aclrtSetCurrentContext(context);
    }
    if (err != ACL_SUCCESS) {
        LOG_ERROR("DvppVpcScaler: aclrtSetCurrentContext failed ({})",
                  static_cast<int>(err));
        return false;
    }

    channel_desc_ = acldvppCreateChannelDesc();
    if (!channel_desc_) {
        LOG_ERROR("DvppVpcScaler: acldvppCreateChannelDesc returned null");
        return false;
    }

    if (vpc_api_.createChannel) {
        err = vpc_api_.createChannel(channel_desc_);
    } else {
        err = acldvppCreateChannel(channel_desc_);
    }
    if (err != ACL_SUCCESS) {
        LOG_ERROR("DvppVpcScaler: createChannel failed ({})", static_cast<int>(err));
        acldvppDestroyChannelDesc(channel_desc_);
        channel_desc_ = nullptr;
        return false;
    }

    err = aclrtCreateStream(&stream_);
    if (err != ACL_SUCCESS) {
        LOG_ERROR("DvppVpcScaler: aclrtCreateStream failed ({})", static_cast<int>(err));
        shutdown();
        return false;
    }

    resize_cfg_ = acldvppCreateResizeConfig();
    if (!resize_cfg_) {
        LOG_ERROR("DvppVpcScaler: acldvppCreateResizeConfig returned null");
        shutdown();
        return false;
    }

    LOG_INFO("DvppVpcScaler: VPC channel ready device={}", device_id_);
    return true;
}

void DvppVpcScaler::shutdown() {
    if (resize_cfg_) {
        acldvppDestroyResizeConfig(resize_cfg_);
        resize_cfg_ = nullptr;
    }
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (channel_desc_) {
        aclError err = ACL_SUCCESS;
        if (vpc_api_.destroyChannel) {
            err = vpc_api_.destroyChannel(channel_desc_);
        } else {
            err = acldvppDestroyChannel(channel_desc_);
        }
        if (err != ACL_SUCCESS) {
            LOG_WARN("DvppVpcScaler: destroyChannel failed ({})", static_cast<int>(err));
        }
        acldvppDestroyChannelDesc(channel_desc_);
        channel_desc_ = nullptr;
    }
}

bool DvppVpcScaler::resize(acldvppPicDesc* input, acldvppPicDesc* output) {
    if (!channel_desc_ || !resize_cfg_ || !stream_ || !input || !output) {
        return false;
    }

    aclError err = ACL_SUCCESS;
    if (vpc_api_.vpcResize) {
        err = vpc_api_.vpcResize(channel_desc_, input, output, resize_cfg_, stream_);
    } else {
        err = acldvppVpcResizeAsync(channel_desc_, input, output, resize_cfg_, stream_);
    }
    if (err != ACL_SUCCESS) {
        LOG_WARN("DvppVpcScaler: vpc resize failed ({})", static_cast<int>(err));
        return false;
    }

    err = aclrtSynchronizeStream(stream_);
    if (err != ACL_SUCCESS) {
        LOG_WARN("DvppVpcScaler: aclrtSynchronizeStream failed ({})", static_cast<int>(err));
        return false;
    }
    return true;
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
