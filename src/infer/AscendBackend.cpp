#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBackend.h"
#include "common/Logger.h"

#include <acl/acl.h>
#include <stdexcept>
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace infer {

// Process-wide ACL refcount definition.
std::atomic<int> AscendBackend::s_acl_refcount_{0};

namespace {

#define ACL_CHECK(call) do { \
    aclError _err = (call); \
    if (_err != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("[ACL] error ") + std::to_string(_err) \
                                 + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// 100002: ACL_ERROR_REPEAT_INITIALIZE (already inited in same process)
// 507008: ACL_ERROR_RT_CONTEXT_NULL_PTR (CANN 6 emits this instead of 100002 on repeat aclInit)
constexpr aclError kAclRepeatInitCode     = static_cast<aclError>(100002);
constexpr aclError kAclRepeatInitCodeCann6 = static_cast<aclError>(507008);

int yoloAnchorCount(int h, int w) {
    return (h / 8) * (w / 8)
         + (h / 16) * (w / 16)
         + (h / 32) * (w / 32);
}

void packBgrToNv12(const Batch& input, uint8_t* dst,
                   int batch_size, int h, int w) {
    // CPU-side conversion for AIPP models that take NV12 (YUV420SP) input.
    // Layout: [Y plane (h*w)] + [UV interleaved (h*w/2)].
    const size_t y_bytes  = static_cast<size_t>(h) * w;
    const size_t uv_bytes = y_bytes / 2;
    const size_t frame_bytes = y_bytes + uv_bytes; // h*w*3/2

    for (int b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(input.frames[b], resized, {w, h});

        cv::Mat i420;
        cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
        const uint8_t* i420_ptr = i420.ptr<uint8_t>(0);

        uint8_t* out = dst + static_cast<size_t>(b) * frame_bytes;

        // Y
        std::memcpy(out, i420_ptr, y_bytes);

        // U and V planes in I420 are quarter size each (h/2 * w/2).
        const uint8_t* u = i420_ptr + y_bytes;
        const uint8_t* v = u + (y_bytes / 4);

        // NV12 interleaves UV.
        uint8_t* uv = out + y_bytes;
        for (size_t i = 0; i < (y_bytes / 4); ++i) {
            uv[2 * i + 0] = u[i];
            uv[2 * i + 1] = v[i];
        }
    }
}

// RAII guard that releases an AscendPooledBuffer on scope exit.
struct SlotGuard {
    AscendBufferPool&   pool;
    AscendPooledBuffer* slot;
    ~SlotGuard() { pool.release(slot); }
};

} // namespace

// ── Lifecycle ──────────────────────────────────────────────────────────────

AscendBackend::AscendBackend() = default;

AscendBackend::~AscendBackend() {
    unloadModel();
}

void AscendBackend::loadModel(const ModelConfig& cfg) {
    device_id_      = cfg.device_id;
    max_batch_size_ = cfg.batch_size;
    input_h_        = cfg.input_shape.height;
    input_w_        = cfg.input_shape.width;
    num_classes_    = cfg.num_classes;

    // Increment process-wide refcount; call aclInit only on the first backend.
    if (s_acl_refcount_.fetch_add(1, std::memory_order_acq_rel) == 0) {
        const aclError init_rc = aclInit(nullptr);
        if (init_rc != ACL_SUCCESS && init_rc != kAclRepeatInitCode && init_rc != kAclRepeatInitCodeCann6) {
            s_acl_refcount_.fetch_sub(1, std::memory_order_release);
            throw std::runtime_error(std::string("[ACL] aclInit error ") + std::to_string(init_rc)
                                     + " at " + __FILE__ + ":" + std::to_string(__LINE__));
        }
        LOG_INFO("AscendBackend: aclInit succeeded (refcount=1)");
    } else {
        LOG_INFO("AscendBackend: reusing ACL runtime (refcount={})",
                 s_acl_refcount_.load(std::memory_order_relaxed));
    }
    ACL_CHECK(aclrtSetDevice(device_id_));
    // CANN 6 does not auto-create a default context; explicit creation is
    // required for correct multi-threaded operation on all CANN versions.
    ACL_CHECK(aclrtCreateContext(&ctx_, device_id_));
    ACL_CHECK(aclrtCreateStream(&stream_));

    for (auto& [bs, path] : cfg.om_paths) {
        uint32_t model_id = 0;
        ACL_CHECK(aclmdlLoadFromFile(path.c_str(), &model_id));
        model_map_[bs] = model_id;
        LOG_INFO("AscendBackend: loaded om {} (batch={})", path, bs);
    }

    if (model_map_.empty()) {
        throw std::runtime_error("AscendBackend: no om_paths configured");
    }

    // Detect AIPP from the first loaded model
    auto detect = aipp_detect_fn_ ? aipp_detect_fn_
                : [this](uint32_t id) { return detectAipp(id); };
    aipp_enabled_ = detect(model_map_.begin()->second);

    const int max_bs = model_map_.rbegin()->first;
    if (aipp_enabled_) {
        // AIPP path: NV12 (YUV420SP) uint8 input.
        // The AIPP config in our exported .om expects NV12 (see docs/ascend-guide.md).
        input_bytes_ = static_cast<size_t>(max_bs) * input_h_ * input_w_ * 3 / 2;
        LOG_INFO("AscendBackend: AIPP enabled, input format=NV12(uint8)");
    } else {
        // CPU preprocess path: CHW float input
        input_bytes_  = static_cast<size_t>(max_bs) * 3 * input_h_ * input_w_ * sizeof(float);
    }
    const int anchors = yoloAnchorCount(input_h_, input_w_);
    output_bytes_ = static_cast<size_t>(max_bs) * (4 + num_classes_) * anchors * sizeof(float);

    buffer_pool_.init(device_id_, ctx_, /*pool_size=*/4, input_bytes_, output_bytes_);

    loaded_ = true;
}

void AscendBackend::unloadModel() {
    if (!loaded_) return;
    buffer_pool_.reset();
    for (auto& [bs, id] : model_map_) {
        aclmdlUnload(id);
    }
    model_map_.clear();
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (ctx_) {
        aclrtDestroyContext(ctx_);
        ctx_ = nullptr;
    }
    // aclrtResetDevice is intentionally omitted: it is a process-wide operation
    // that destroys all ACL resources on the device, including those held by
    // sibling AscendBackend instances on the same device_id.
    //
    // Decrement refcount; call aclFinalize only when the last backend stops.
    if (s_acl_refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        aclFinalize();
        LOG_INFO("AscendBackend: aclFinalize called (last backend stopped)");
    }
    loaded_ = false;
}

// ── Private helpers ────────────────────────────────────────────────────────

uint32_t AscendBackend::selectModel(int batch_size) const {
    // Choose the smallest compiled batch that can accommodate the request.
    // The infer() path pads input to this batch and slices outputs back.
    if (model_map_.empty()) {
        throw std::runtime_error("AscendBackend: no models loaded");
    }
    auto it = model_map_.lower_bound(batch_size);
    if (it != model_map_.end()) return it->second;
    return model_map_.rbegin()->second;
}

void AscendBackend::preprocessCPU(const Batch& input, float* dst,
                                   int batch_size, int h, int w) {
    const float inv_255 = 1.0f / 255.0f;
    for (int b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(input.frames[b], resized, {w, h});
        float* ch_r = dst + b * 3 * h * w;
        float* ch_g = ch_r + h * w;
        float* ch_b = ch_g + h * w;
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x) {
                ch_b[y * w + x] = row[x * 3 + 0] * inv_255;
                ch_g[y * w + x] = row[x * 3 + 1] * inv_255;
                ch_r[y * w + x] = row[x * 3 + 2] * inv_255;
            }
        }
    }
}

void AscendBackend::packBgrUint8(const Batch& input, uint8_t* dst,
                                  int batch_size, int h, int w) {
    // AIPP path: resize to (w,h), then pack as HWC BGR uint8.
    // AIPP hardware handles CHW conversion and normalization internally.
    for (int b = 0; b < batch_size; ++b) {
        cv::Mat resized;
        cv::resize(input.frames[b], resized, {w, h});
        // resized is HWC BGR uint8 — copy directly
        const size_t frame_bytes = static_cast<size_t>(h) * w * 3;
        std::memcpy(dst + b * frame_bytes, resized.data, frame_bytes);
    }
}

bool AscendBackend::detectAipp(uint32_t model_id) const {
    // aclmdlGetFirstAippInfo returns ACL_SUCCESS when the model has AIPP.
    // ACL_ERROR_GE_AIPP_NOT_EXIST (148034) means no AIPP operator.
    aclAippInfo aipp_info{};
    return aclmdlGetFirstAippInfo(model_id, 0, &aipp_info) == ACL_SUCCESS;
}

size_t AscendBackend::expectedOutputBytesForBatchForTest(int batch_size) const {
    return static_cast<size_t>(batch_size)
        * (4 + num_classes_)
        * yoloAnchorCount(input_h_, input_w_)
        * sizeof(float);
}

// ── infer() ────────────────────────────────────────────────────────────────

void AscendBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) throw std::runtime_error("AscendBackend: model not loaded");
    std::lock_guard<std::mutex> lk(infer_mu_);
    ACL_CHECK(aclrtSetCurrentContext(ctx_));

    const int req_bs = input.size();
    // Round up to the smallest compiled batch and pad inputs accordingly.
    auto it = model_map_.lower_bound(req_bs);
    if (it == model_map_.end()) {
        throw std::runtime_error("AscendBackend: request batch_size exceeds compiled max batch");
    }
    const int      model_bs = it->first;
    const uint32_t model_id = it->second;

    const size_t out_bytes_model =
        static_cast<size_t>(model_bs) * (4 + num_classes_) * yoloAnchorCount(input_h_, input_w_) * sizeof(float);
    const size_t out_bytes_req =
        static_cast<size_t>(req_bs) * (4 + num_classes_) * yoloAnchorCount(input_h_, input_w_) * sizeof(float);

    // Acquire pre-allocated slot for the output buffer (exception-safe release).
    AscendPooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) throw std::runtime_error("AscendBackend: buffer pool timed out");
    SlotGuard guard{buffer_pool_, slot};

    auto do_memcpy = memcpy_fn_ ? memcpy_fn_
        : [](void* dst, size_t dst_size, const void* src, size_t src_size,
             aclrtMemcpyKind kind) -> aclError {
            return aclrtMemcpy(dst, dst_size, src, src_size, kind);
        };

    // ── Select input pointer and compute in_bytes ─────────────────────────
    //
    // Three paths (mutually exclusive, checked in priority order):
    //
    //  A. Zero-copy  — DVPP NV12 frames already in HBM + AIPP model
    //       batch=1  → DVPP device ptr forwarded directly (no copy at all)
    //       batch>1  → frames D2D-concatenated into a temporary device buffer
    //
    //  B. AIPP / CPU BGR  — CPU-side packBgrUint8 into pre-allocated slot
    //
    //  C. CPU float CHW   — CPU-side preprocessCPU into pre-allocated slot

    void*  input_ptr  = slot->input_device;  // default: pre-allocated slot
    size_t in_bytes   = 0;
    void*  tmp_device = nullptr;             // non-null → freed by TmpDevGuard

    // RAII guard for the temporary D2D concatenation buffer (path A, batch>1).
    // Holds a reference to tmp_device so it sees the value set after alloc.
    // Freed before the slot is released (correct destruction order on unwind).
    struct TmpDevGuard {
        void**       ptr_ref;
        AclDevFreeFn free_fn;
        ~TmpDevGuard() {
            if (!*ptr_ref) return;
            if (free_fn) { free_fn(*ptr_ref); } else { aclrtFree(*ptr_ref); }
        }
    } tmp_guard{&tmp_device, dev_free_fn_};

#ifdef BUILD_ASCEND_BACKEND
    if (aipp_enabled_ && input.is_ascend && !input.ascend_frames.empty()) {
        // Path A: zero-copy DVPP → AIPP
        const auto& f0 = input.ascend_frames[0];
        const size_t frame_nv12 =
            static_cast<size_t>(f0.width) * static_cast<size_t>(f0.height) * 3 / 2;
        in_bytes = static_cast<size_t>(model_bs) * frame_nv12;

        if (req_bs == 1 && model_bs == 1) {
            // True zero-copy: forward DVPP device ptr directly to ACL dataset.
            input_ptr = f0.yuv_device;
        } else {
            // Need a contiguous buffer sized for model_bs with padding.
            // TmpDevGuard is already live so any throw below is exception-safe.
            auto alloc_fn = dev_alloc_fn_ ? dev_alloc_fn_
                : [](void** ptr, size_t sz) -> aclError {
                    return aclrtMalloc(ptr, sz, ACL_MEM_MALLOC_HUGE_FIRST);
                };
            ACL_CHECK(alloc_fn(&tmp_device, in_bytes));

            for (int b = 0; b < req_bs; ++b) {
                const auto& fb  = input.ascend_frames[b];
                void*       dst = static_cast<char*>(tmp_device)
                                  + static_cast<ptrdiff_t>(b) * frame_nv12;
                ACL_CHECK(do_memcpy(dst, frame_nv12, fb.yuv_device, frame_nv12,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE));
            }
            if (model_bs > req_bs) {
                void* pad_dst = static_cast<char*>(tmp_device)
                                + static_cast<ptrdiff_t>(req_bs) * frame_nv12;
                const size_t pad_bytes = static_cast<size_t>(model_bs - req_bs) * frame_nv12;
                ACL_CHECK(aclrtMemset(pad_dst, pad_bytes, 0, pad_bytes));
            }
            input_ptr = tmp_device;
        }
    } else
#endif
    if (aipp_enabled_) {
        // Path B: CPU BGR → NV12 (YUV420SP) uint8.
        // slot->input_device is NPU HBM memory, so we must stage on host first.
        in_bytes = static_cast<size_t>(model_bs) * input_h_ * input_w_ * 3 / 2;
        std::vector<uint8_t> host_input(in_bytes, 0);
        packBgrToNv12(input, host_input.data(), req_bs, input_h_, input_w_);
        ACL_CHECK(do_memcpy(slot->input_device, input_bytes_, host_input.data(), in_bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE));
        input_ptr = slot->input_device;
    } else {
        // Path C: CPU float CHW, no AIPP.
        // Preprocess on host, then copy into device input buffer.
        in_bytes = static_cast<size_t>(model_bs) * 3 * input_h_ * input_w_ * sizeof(float);
        std::vector<float> host_input(in_bytes / sizeof(float), 0.0f);
        preprocessCPU(input, host_input.data(), req_bs, input_h_, input_w_);
        ACL_CHECK(do_memcpy(slot->input_device, input_bytes_, host_input.data(), in_bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE));
        input_ptr = slot->input_device;
    }

    // ── Build ACL datasets and run async inference ────────────────────────
    aclDataBuffer* in_buf  = aclCreateDataBuffer(input_ptr, in_bytes);
    aclDataBuffer* out_buf = aclCreateDataBuffer(slot->output_device, out_bytes_model);
    if (!in_buf || !out_buf) {
        if (in_buf) aclDestroyDataBuffer(in_buf);
        if (out_buf) aclDestroyDataBuffer(out_buf);
        throw std::runtime_error("AscendBackend: aclCreateDataBuffer returned null");
    }
    aclmdlDataset* in_set  = aclmdlCreateDataset();
    aclmdlDataset* out_set = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(in_set, in_buf);
    aclmdlAddDatasetBuffer(out_set, out_buf);

    struct DatasetGuard {
        aclmdlDataset* in_s; aclmdlDataset* out_s;
        aclDataBuffer* in_b; aclDataBuffer* out_b;
        ~DatasetGuard() {
            aclmdlDestroyDataset(in_s);
            aclmdlDestroyDataset(out_s);
            aclDestroyDataBuffer(in_b);
            aclDestroyDataBuffer(out_b);
        }
    } ds_guard{in_set, out_set, in_buf, out_buf};

    auto exec_fn = execute_async_fn_ ? execute_async_fn_
        : [](uint32_t m, aclmdlDataset* i, aclmdlDataset* o, aclrtStream s) -> aclError {
            return aclmdlExecuteAsync(m, i, o, s);
        };
    ACL_CHECK(exec_fn(model_id, in_set, out_set, stream_));

    auto sync_fn = sync_stream_fn_ ? sync_stream_fn_
        : [](aclrtStream s) -> aclError { return aclrtSynchronizeStream(s); };
    auto timed_sync_fn = sync_stream_with_timeout_fn_ ? sync_stream_with_timeout_fn_
        : [](aclrtStream s, uint32_t timeout_ms) -> aclError {
            return aclrtSynchronizeStreamWithTimeout(s, timeout_ms);
        };
    if (infer_timeout_ms_ > 0) {
        ACL_CHECK(timed_sync_fn(stream_, infer_timeout_ms_));
    } else {
        ACL_CHECK(sync_fn(stream_));
    }

    // D2H: copy NPU output back to host (async + sync)
    auto memcpy_async_fn = memcpy_async_fn_ ? memcpy_async_fn_
        : [](void* dst, size_t dst_size, const void* src, size_t src_size,
             aclrtMemcpyKind kind, aclrtStream stream) -> aclError {
            return aclrtMemcpyAsync(dst, dst_size, src, src_size, kind, stream);
        };
    std::vector<float> host_out(out_bytes_model / sizeof(float));
    ACL_CHECK(memcpy_async_fn(host_out.data(), out_bytes_model,
                              slot->output_device, out_bytes_model,
                              ACL_MEMCPY_DEVICE_TO_HOST, stream_));
    if (infer_timeout_ms_ > 0) {
        ACL_CHECK(timed_sync_fn(stream_, infer_timeout_ms_));
    } else {
        ACL_CHECK(sync_fn(stream_));
    }

    output.assign(host_out.begin(),
                  host_out.begin() + static_cast<ptrdiff_t>(out_bytes_req / sizeof(float)));
}

// ── Test entry point ────────────────────────────────────────────────────────

void AscendBackend::initPoolForTest(int pool_size,
                                     size_t input_bytes, size_t output_bytes) {
    input_bytes_  = input_bytes;
    output_bytes_ = output_bytes;
    buffer_pool_.initForTest(pool_size, input_bytes, output_bytes);
}

void AscendBackend::inferWithStubs(const Batch& input, std::vector<float>& output) {
    // Simplified infer path used by tests: skips real ACL dataset calls.
    // Exercises pool acquire/release + preprocess branch selection + exec/sync ordering.
    //
    // Supported paths (same branches as infer()):
    //   A. Zero-copy DVPP  — input.is_ascend && aipp_enabled_
    //   B. CPU BGR / AIPP  — aipp_enabled_, CPU frames
    //   C. CPU float CHW   — no AIPP, CPU frames

#ifdef BUILD_ASCEND_BACKEND
    const bool ascend_path = aipp_enabled_ && input.is_ascend
                             && !input.ascend_frames.empty();
#else
    const bool ascend_path = false;
#endif
    if (!ascend_path && input.frames.empty()) return;

    AscendPooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) throw std::runtime_error("AscendBackend: buffer pool timed out");
    SlotGuard guard{buffer_pool_, slot};

    auto do_memcpy = memcpy_fn_ ? memcpy_fn_
        : [](void*, size_t, const void*, size_t, aclrtMemcpyKind) -> aclError {
            return ACL_SUCCESS;
        };

    // ── Path A: zero-copy DVPP ─────────────────────────────────────────────
    void* tmp_device = nullptr;

    // RAII cleanup for tmp buffer (path A, batch>1).
    // References tmp_device so it sees the value set after alloc.
    struct TmpDevGuard {
        void**       ptr_ref;
        AclDevFreeFn free_fn;
        ~TmpDevGuard() {
            if (!*ptr_ref) return;
            if (free_fn) { free_fn(*ptr_ref); } else { ::operator delete(*ptr_ref); }
        }
    } tmp_guard{&tmp_device, dev_free_fn_};

#ifdef BUILD_ASCEND_BACKEND
    if (ascend_path) {
        const int bs = static_cast<int>(input.ascend_frames.size());
        const auto& f0 = input.ascend_frames[0];
        const size_t frame_nv12 =
            static_cast<size_t>(f0.width) * static_cast<size_t>(f0.height) * 3 / 2;

        if (bs > 1) {
            // Multi-frame: alloc tmp buffer + D2D memcpy per frame.
            // TmpDevGuard is already live so any throw below is exception-safe.
            const size_t total = static_cast<size_t>(bs) * frame_nv12;
            auto alloc_fn = dev_alloc_fn_ ? dev_alloc_fn_
                : [](void** ptr, size_t sz) -> aclError {
                    *ptr = ::operator new(sz);
                    return ACL_SUCCESS;
                };
            ACL_CHECK(alloc_fn(&tmp_device, total));

            for (int b = 0; b < bs; ++b) {
                const auto& fb  = input.ascend_frames[b];
                void*       dst = static_cast<char*>(tmp_device)
                                  + static_cast<ptrdiff_t>(b) * frame_nv12;
                ACL_CHECK(do_memcpy(dst, frame_nv12, fb.yuv_device, frame_nv12,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE));
            }
        }
        // batch=1: no copy, DVPP ptr used directly (no-op in stub mode)
    }
#endif

    // ── Fake dataset handles (nullptr is fine with stub execute fn) ────────
    aclmdlDataset* in_set  = nullptr;
    aclmdlDataset* out_set = nullptr;

    auto exec_fn = execute_async_fn_ ? execute_async_fn_
        : [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            return ACL_SUCCESS;
        };
    ACL_CHECK(exec_fn(0u, in_set, out_set, stream_));

    auto sync_fn = sync_stream_fn_ ? sync_stream_fn_
        : [](aclrtStream) -> aclError { return ACL_SUCCESS; };
    if (infer_timeout_ms_ > 0 && sync_stream_with_timeout_fn_) {
        ACL_CHECK(sync_stream_with_timeout_fn_(stream_, infer_timeout_ms_));
    } else {
        ACL_CHECK(sync_fn(stream_));
    }

    auto memcpy_async_fn = memcpy_async_fn_ ? memcpy_async_fn_
        : [&do_memcpy](void* dst, size_t dst_size, const void* src, size_t src_size,
                       aclrtMemcpyKind kind, aclrtStream) -> aclError {
            return do_memcpy(dst, dst_size, src, src_size, kind);
        };
    output.resize(output_bytes_ / sizeof(float));
    ACL_CHECK(memcpy_async_fn(output.data(), output_bytes_,
                              slot->output_device, output_bytes_,
                              ACL_MEMCPY_DEVICE_TO_HOST, stream_));
    if (infer_timeout_ms_ > 0 && sync_stream_with_timeout_fn_) {
        ACL_CHECK(sync_stream_with_timeout_fn_(stream_, infer_timeout_ms_));
    } else {
        ACL_CHECK(sync_fn(stream_));
    }
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
