#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendProcessRuntime.h"

#include "common/Logger.h"

#include <acl/acl.h>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace infer {

namespace {

std::atomic<int> g_acl_refcount{0};
std::atomic<bool> g_acl_runtime_initialized{false};
std::once_flag    g_acl_atexit_registered;

static void aclProcessAtExit() noexcept {
    if (g_acl_runtime_initialized.load(std::memory_order_acquire)) {
        aclFinalize();
        g_acl_runtime_initialized.store(false, std::memory_order_release);
    }
}

static void registerAclProcessTeardown() {
    std::call_once(g_acl_atexit_registered, [] { std::atexit(aclProcessAtExit); });
}

constexpr aclError kAclRepeatInitCode      = static_cast<aclError>(100002);
constexpr aclError kAclRepeatInitCodeCann6 = static_cast<aclError>(507008);

} // namespace

void AscendProcessRuntime::acquire() {
    const int prev_ref = g_acl_refcount.fetch_add(1, std::memory_order_acq_rel);
    if (prev_ref != 0) {
        LOG_INFO("AscendProcessRuntime: reusing ACL runtime (refcount={})",
                 g_acl_refcount.load(std::memory_order_relaxed));
        return;
    }
    if (!g_acl_runtime_initialized.load(std::memory_order_acquire)) {
        const aclError init_rc = aclInit(nullptr);
        if (init_rc != ACL_SUCCESS && init_rc != kAclRepeatInitCode && init_rc != kAclRepeatInitCodeCann6) {
            g_acl_refcount.fetch_sub(1, std::memory_order_release);
            throw std::runtime_error(std::string("[ACL] aclInit error ") + std::to_string(init_rc));
        }
        g_acl_runtime_initialized.store(true, std::memory_order_release);
        registerAclProcessTeardown();
        LOG_INFO("AscendProcessRuntime: aclInit succeeded (refcount=1)");
    } else {
        LOG_INFO("AscendProcessRuntime: reusing ACL runtime after hot unload (refcount=1)");
    }
}

void AscendProcessRuntime::release() {
    if (g_acl_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        LOG_INFO("AscendProcessRuntime: last consumer stopped (ACL runtime kept until process exit)");
    }
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
