#include "decoder/DecoderFactory.h"
#include "decoder/YOLOv5Decoder.h"
#include "decoder/YOLOv8Decoder.h"
#include "decoder/YOLO11Decoder.h"
#include "decoder/YOLO26Decoder.h"
#include "decoder/ClassifierDecoder.h"
#include "common/Logger.h"
#include <stdexcept>

namespace infer {

std::unique_ptr<IYOLODecoder> createDecoder(const ModelConfig& cfg) {
    if (cfg.model_type == ModelType::Classifier) {
        return std::make_unique<ClassifierDecoder>(cfg);
    }

    switch (cfg.version) {
    case YOLOVersion::v5:
        // Many Ascend-exported YOLOv5u/YOLOv5s models emit YOLOv8-style
        // tensors (8400, 4+classes) instead of legacy v5 (25200, 5+classes).
        // Keep older CPU/CUDA behavior unchanged, but prefer v8 decoder on
        // Ascend to avoid out-of-bounds decode on 8400-anchor outputs.
        if (cfg.backend == DeviceType::Ascend) {
            LOG_WARN("createDecoder: model version=v5 with backend=ascend; using YOLOv8 decoder for 8400-anchor output layout");
            return std::make_unique<YOLOv8Decoder>(cfg.num_classes);
        }
        return std::make_unique<YOLOv5Decoder>(cfg.num_classes);
    case YOLOVersion::v8:
        return std::make_unique<YOLOv8Decoder>(cfg.num_classes);
    case YOLOVersion::v11:
        return std::make_unique<YOLO11Decoder>(cfg.num_classes);
    case YOLOVersion::v26:
        return std::make_unique<YOLO26Decoder>(cfg.num_classes);
    default:
        throw std::runtime_error("createDecoder: unknown YOLOVersion");
    }
}

} // namespace infer
