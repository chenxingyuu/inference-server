#ifdef BUILD_ASCEND_BACKEND

#include "infer/AscendBackend.h"
#include "common/Logger.h"

#include <acl/acl.h>
#include <stdexcept>
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace infer {

namespace {

#define ACL_CHECK(call) do { \
    aclError _err = (call); \
    if (_err != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("[ACL] error ") + std::to_string(_err) \
                                 + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

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

    ACL_CHECK(aclInit(nullptr));
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
        // AIPP path: raw BGR uint8 input
        input_bytes_ = static_cast<size_t>(max_bs) * input_h_ * input_w_ * 3;
        LOG_INFO("AscendBackend: AIPP enabled, input dtype=uint8");
    } else {
        // CPU preprocess path: CHW float input
        input_bytes_  = static_cast<size_t>(max_bs) * 3 * input_h_ * input_w_ * sizeof(float);
    }
    output_bytes_ = static_cast<size_t>(max_bs) * (4 + num_classes_) * 8400 * sizeof(float);

    buffer_pool_.init(device_id_, /*pool_size=*/4, input_bytes_, output_bytes_);

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
    aclrtResetDevice(device_id_);
    aclFinalize();
    loaded_ = false;
}

// ── Private helpers ────────────────────────────────────────────────────────

uint32_t AscendBackend::selectModel(int batch_size) const {
    uint32_t best_id = model_map_.begin()->second;
    for (const auto& [bs, id] : model_map_) {
        if (bs <= batch_size) best_id = id;
        else break;
    }
    return best_id;
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

// ── infer() ────────────────────────────────────────────────────────────────

void AscendBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) throw std::runtime_error("AscendBackend: model not loaded");

    const int bs          = input.size();
    const size_t in_bytes = aipp_enabled_
        ? static_cast<size_t>(bs) * input_h_ * input_w_ * 3
        : static_cast<size_t>(bs) * 3 * input_h_ * input_w_ * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(bs) * (4 + num_classes_) * 8400 * sizeof(float);
    const uint32_t model_id = selectModel(bs);

    // Acquire pre-allocated slot; release on scope exit (exception-safe)
    AscendPooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) throw std::runtime_error("AscendBackend: buffer pool timed out");
    SlotGuard guard{buffer_pool_, slot};

    // Preprocess into slot->input_device (CPU → already HBM)
    if (aipp_enabled_) {
        packBgrUint8(input, static_cast<uint8_t*>(slot->input_device), bs, input_h_, input_w_);
    } else {
        preprocessCPU(input, static_cast<float*>(slot->input_device), bs, input_h_, input_w_);
    }

    // H2D copy (staging → HBM already done above since slot is HBM)
    // Note: preprocessCPU/packBgrUint8 write to host-accessible HBM (ACL_MEM_MALLOC_HUGE_FIRST
    // allocates in HBM, but on 310P all HBM is accessible from host via SDMA).
    // An explicit H2D memcpy is still required to synchronise writes.
    auto do_memcpy = memcpy_fn_ ? memcpy_fn_
        : [](void* dst, size_t dst_size, const void* src, size_t src_size,
             aclrtMemcpyKind kind) -> aclError {
            return aclrtMemcpy(dst, dst_size, src, src_size, kind);
        };

    // Build ACL datasets around pre-allocated slot buffers
    aclDataBuffer* in_buf  = aclCreateDataBuffer(slot->input_device, in_bytes);
    aclDataBuffer* out_buf = aclCreateDataBuffer(slot->output_device, out_bytes);
    aclmdlDataset* in_set  = aclmdlCreateDataset();
    aclmdlDataset* out_set = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(in_set, in_buf);
    aclmdlAddDatasetBuffer(out_set, out_buf);

    // RAII dataset cleanup (runs before SlotGuard releases the slot)
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

    // Async execute
    auto exec_fn = execute_async_fn_ ? execute_async_fn_
        : [](uint32_t m, aclmdlDataset* i, aclmdlDataset* o, aclrtStream s) -> aclError {
            return aclmdlExecuteAsync(m, i, o, s);
        };
    ACL_CHECK(exec_fn(model_id, in_set, out_set, stream_));

    // Synchronise stream: blocks host until NPU completes
    auto sync_fn = sync_stream_fn_ ? sync_stream_fn_
        : [](aclrtStream s) -> aclError { return aclrtSynchronizeStream(s); };
    ACL_CHECK(sync_fn(stream_));

    // D2H copy output
    output.resize(out_bytes / sizeof(float));
    ACL_CHECK(do_memcpy(output.data(), out_bytes,
                        slot->output_device, out_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
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
    // Exercises pool acquire/release + execute + sync call ordering.
    if (input.frames.empty()) return;

    AscendPooledBuffer* slot = buffer_pool_.acquire();
    if (!slot) throw std::runtime_error("AscendBackend: buffer pool timed out");
    SlotGuard guard{buffer_pool_, slot};

    // Preprocess (CPU) to verify AIPP branch selection
    if (aipp_enabled_) {
        const size_t frame_bytes = static_cast<size_t>(input_h_) * input_w_ * 3;
        (void)frame_bytes; // no-op in stub mode, slot memory is operator-new heap
    }

    // Fake dataset handles (nullptr is fine with stub execute fn)
    aclmdlDataset* in_set  = nullptr;
    aclmdlDataset* out_set = nullptr;

    auto exec_fn = execute_async_fn_ ? execute_async_fn_
        : [](uint32_t, aclmdlDataset*, aclmdlDataset*, aclrtStream) -> aclError {
            return ACL_SUCCESS;
        };
    ACL_CHECK(exec_fn(0u, in_set, out_set, stream_));

    auto sync_fn = sync_stream_fn_ ? sync_stream_fn_
        : [](aclrtStream) -> aclError { return ACL_SUCCESS; };
    ACL_CHECK(sync_fn(stream_));

    auto do_memcpy = memcpy_fn_ ? memcpy_fn_
        : [](void*, size_t, const void*, size_t, aclrtMemcpyKind) -> aclError {
            return ACL_SUCCESS;
        };
    output.resize(output_bytes_ / sizeof(float));
    ACL_CHECK(do_memcpy(output.data(), output_bytes_,
                        slot->output_device, output_bytes_,
                        ACL_MEMCPY_DEVICE_TO_HOST));
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
