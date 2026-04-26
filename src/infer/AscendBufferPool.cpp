#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBufferPool.h"
#include "common/Logger.h"

#include <acl/acl.h>
#include <stdexcept>
#include <thread>
#include <chrono>

#define ACL_POOL_CHECK(call) do { \
    aclError _err = (call); \
    if (_err != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("[AscendBufferPool ACL] error ") \
                                 + std::to_string(_err)); \
    } \
} while(0)

namespace infer {

// ── AscendPooledBuffer move constructor ────────────────────────────────────

AscendPooledBuffer::AscendPooledBuffer(AscendPooledBuffer&& o) noexcept
    : input_device(o.input_device)
    , output_device(o.output_device)
    , in_use(o.in_use.load(std::memory_order_relaxed))
    , idx(o.idx) {
    o.input_device  = nullptr;
    o.output_device = nullptr;
    o.in_use.store(false, std::memory_order_relaxed);
}

// ── AscendBufferPool ───────────────────────────────────────────────────────

AscendBufferPool::AscendBufferPool(MemoryChecker checker)
    : memory_checker_(std::move(checker)) {
    if (!memory_checker_) {
        memory_checker_ = [this]() -> bool {
            size_t free_bytes = 0, total_bytes = 0;
            if (aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) != ACL_SUCCESS
                || total_bytes == 0) {
                return true; // query failed — allow through
            }
            const float used_ratio = static_cast<float>(total_bytes - free_bytes)
                                     / static_cast<float>(total_bytes);
            return used_ratio < kOomGuardRatio;
        };
    }
}

bool AscendBufferPool::isSafe() const {
    return memory_checker_ ? memory_checker_() : true;
}

void AscendBufferPool::init(int device_id, int pool_size,
                             size_t input_bytes, size_t output_bytes) {
    device_id_  = device_id;
    test_mode_  = false;
    ACL_POOL_CHECK(aclrtSetDevice(device_id_));

    slots_.resize(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        auto& s = slots_[i];
        s.idx   = i;
        s.in_use.store(false, std::memory_order_relaxed);
        ACL_POOL_CHECK(aclrtMalloc(&s.input_device,  input_bytes,  ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_POOL_CHECK(aclrtMalloc(&s.output_device, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    LOG_INFO("AscendBufferPool: allocated {} slots ({} + {} bytes each)",
             pool_size, input_bytes, output_bytes);
}

void AscendBufferPool::initForTest(int pool_size,
                                    size_t input_bytes, size_t output_bytes) {
    test_mode_ = true;
    slots_.resize(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        auto& s = slots_[i];
        s.idx   = i;
        s.in_use.store(false, std::memory_order_relaxed);
        s.input_device  = ::operator new(input_bytes);
        s.output_device = ::operator new(output_bytes);
    }
}

void AscendBufferPool::reset() {
    if (slots_.empty()) return;

    for (auto& s : slots_) {
        if (test_mode_) {
            ::operator delete(s.input_device);
            ::operator delete(s.output_device);
        } else {
            if (s.input_device)  { aclrtFree(s.input_device);  }
            if (s.output_device) { aclrtFree(s.output_device); }
        }
        s.input_device  = nullptr;
        s.output_device = nullptr;
        s.in_use.store(false, std::memory_order_relaxed);
    }
    slots_.clear();
}

AscendPooledBuffer* AscendBufferPool::acquire() {
    if (!isSafe()) {
        throw AscendMemoryPressureException("NPU HBM memory usage above safety threshold");
    }

    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(5);

    while (std::chrono::steady_clock::now() < deadline) {
        for (auto& s : slots_) {
            bool expected = false;
            if (s.in_use.compare_exchange_weak(expected, true,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
                return &s;
            }
        }
        std::this_thread::yield();
    }

    LOG_WARN("AscendBufferPool: acquire() timed out after 5ms — pool exhausted");
    return nullptr;
}

void AscendBufferPool::release(AscendPooledBuffer* buf) {
    if (!buf) return;
    buf->in_use.store(false, std::memory_order_release);
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
