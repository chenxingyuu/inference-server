#pragma once

#ifdef BUILD_ASCEND_BACKEND

namespace infer {

// Process-wide AscendCL runtime (aclInit / atexit aclFinalize) with refcounting.
// Used by AscendBackend, AscendVencFfmpegMuxWriter, and any code that needs ACL
// without loading a model.
class AscendProcessRuntime {
public:
    static void acquire();
    static void release();

    AscendProcessRuntime() = delete;
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
