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
    aclError err = (call); \
    if (err != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("[ACL] error ") + std::to_string(err) \
                                 + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)
} // namespace

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

    const int max_bs = model_map_.rbegin()->first;
    const size_t in_elems  = max_bs * 3 * input_h_ * input_w_;
    const size_t out_elems = max_bs * (4 + num_classes_) * 8400;
    input_staging_.resize(in_elems);
    output_staging_.resize(out_elems);

    loaded_ = true;
}

uint32_t AscendBackend::selectModel(int batch_size) const {
    // Round down to the largest available batch size <= requested
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

void AscendBackend::infer(const Batch& input, std::vector<float>& output) {
    if (!loaded_) throw std::runtime_error("AscendBackend: model not loaded");
    const int bs = input.size();

    preprocessCPU(input, input_staging_.data(), bs, input_h_, input_w_);
    uint32_t model_id = selectModel(bs);

    // Build input dataset
    size_t in_bytes = bs * 3 * input_h_ * input_w_ * sizeof(float);
    void*  dev_in   = nullptr;
    ACL_CHECK(aclrtMalloc(&dev_in, in_bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK(aclrtMemcpy(dev_in, in_bytes,
                          input_staging_.data(), in_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    aclDataBuffer* in_buf  = aclCreateDataBuffer(dev_in, in_bytes);
    aclmdlDataset* in_set  = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(in_set, in_buf);

    // Build output dataset
    size_t out_bytes = bs * (4 + num_classes_) * 8400 * sizeof(float);
    void*  dev_out   = nullptr;
    ACL_CHECK(aclrtMalloc(&dev_out, out_bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* out_buf = aclCreateDataBuffer(dev_out, out_bytes);
    aclmdlDataset* out_set = aclmdlCreateDataset();
    aclmdlAddDatasetBuffer(out_set, out_buf);

    ACL_CHECK(aclmdlExecute(model_id, in_set, out_set));

    output.resize(out_bytes / sizeof(float));
    ACL_CHECK(aclrtMemcpy(output.data(), out_bytes,
                          dev_out, out_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    // Cleanup
    aclmdlDestroyDataset(in_set);
    aclmdlDestroyDataset(out_set);
    aclDestroyDataBuffer(in_buf);
    aclDestroyDataBuffer(out_buf);
    aclrtFree(dev_in);
    aclrtFree(dev_out);
}

void AscendBackend::unloadModel() {
    if (!loaded_) return;
    for (auto& [bs, id] : model_map_) {
        aclmdlUnload(id);
    }
    model_map_.clear();
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    aclrtResetDevice(device_id_);
    aclFinalize();
    loaded_ = false;
}

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
