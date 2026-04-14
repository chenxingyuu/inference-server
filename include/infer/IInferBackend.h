#pragma once

#include "common/Types.h"
#include "common/Config.h"
#include <vector>

namespace infer {

class IInferBackend {
public:
    virtual ~IInferBackend() = default;

    // Load model from ModelConfig. Throws on failure.
    virtual void loadModel(const ModelConfig& cfg) = 0;

    // Run inference on a batch of preprocessed frames.
    // output: flat float buffer, layout depends on model version:
    //   YOLOv5/v8/v11: [batch, num_predictions, 5+num_classes]
    //   (populated by backend, resized as needed)
    virtual void infer(const Batch& input, std::vector<float>& output) = 0;

    // Release GPU/NPU resources
    virtual void unloadModel() = 0;

    virtual int          maxBatchSize() const = 0;
    virtual DeviceType   deviceType()   const = 0;
    virtual bool         isLoaded()     const = 0;
};

} // namespace infer
