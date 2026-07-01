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
