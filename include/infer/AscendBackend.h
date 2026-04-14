#pragma once

#ifdef BUILD_ASCEND_BACKEND

#include "infer/IInferBackend.h"
#include <map>
#include <vector>
#include <array>

// Forward-declare ACL types
typedef void* aclrtStream;
typedef unsigned int aclmdlID;
typedef void  aclmdlDesc;
typedef void  aclDataBuffer;
typedef void  aclmdlDataset;

namespace infer {

// Supported batch sizes for pre-compiled .om files
static constexpr std::array<int, 4> kAscendBatchSizes = {1, 4, 8, 16};

class AscendBackend : public IInferBackend {
public:
    AscendBackend();
    ~AscendBackend() override;

    void loadModel(const ModelConfig& cfg) override;
    void infer(const Batch& input, std::vector<float>& output) override;
    void unloadModel() override;

    int        maxBatchSize() const override { return max_batch_size_; }
    DeviceType deviceType()   const override { return DeviceType::Ascend; }
    bool       isLoaded()     const override { return loaded_; }

private:
    // Select the closest .om for the requested batch size (rounds down)
    aclmdlID selectModel(int batch_size) const;

    void preprocessCPU(const Batch& input, float* dst, int batch_size,
                       int h, int w);

    int  max_batch_size_{16};
    int  input_h_{640};
    int  input_w_{640};
    int  num_classes_{80};
    bool loaded_{false};
    int  device_id_{0};

    // batch_size → model_id
    std::map<int, aclmdlID> model_map_;

    aclrtStream   stream_{nullptr};
    std::vector<float> input_staging_;
    std::vector<float> output_staging_;
};

} // namespace infer

#endif // BUILD_ASCEND_BACKEND
