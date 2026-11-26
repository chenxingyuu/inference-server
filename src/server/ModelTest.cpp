#include "server/ModelTest.h"

#include "decoder/ClassifierDecoder.h"
#include "decoder/DecoderFactory.h"
#include "pipeline/stages/DetectionOverlay.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavutil/base64.h>
}

namespace infer {
namespace {

constexpr size_t kMaxPreviewResults = 300;

std::string encodePreview(const cv::Mat& image) {
    constexpr int max_side = 1600;
    cv::Mat preview = image;
    const int longest_side = std::max(image.cols, image.rows);
    if (longest_side > max_side) {
        const double scale = static_cast<double>(max_side) / longest_side;
        cv::resize(image, preview, {std::max(1, cvRound(image.cols * scale)),
                                    std::max(1, cvRound(image.rows * scale))},
                   0, 0, cv::INTER_AREA);
    }
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(".jpg", preview, jpeg, {cv::IMWRITE_JPEG_QUALITY, 82})) {
        throw std::runtime_error("could not encode the image preview");
    }
    std::string encoded(AV_BASE64_SIZE(jpeg.size()), '\0');
    if (!av_base64_encode(encoded.data(), encoded.size(), jpeg.data(), jpeg.size())) {
        throw std::runtime_error("could not encode the image preview");
    }
    encoded.pop_back(); // av_base64_encode writes one trailing null.
    return encoded;
}

} // namespace

nlohmann::json evaluateModelImage(const ModelConfig& model, IInferBackend& backend, const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3 || image.cols > kModelTestMaxImageSide ||
        image.rows > kModelTestMaxImageSide) {
        throw std::invalid_argument("invalid or oversized image");
    }
    Batch batch;
    const int batch_size = backend.minBatchSize();
    if (batch_size < 1 || batch_size > backend.maxBatchSize()) {
        throw std::runtime_error("backend cannot run the model with a supported batch");
    }
    batch.frames.assign(batch_size, image);
    batch.metas.resize(batch_size);
    for (auto& meta : batch.metas) {
        meta.orig_width = image.cols;
        meta.orig_height = image.rows;
    }

    std::vector<float> output;
    backend.infer(batch, output);
    if (output.empty()) throw std::runtime_error("model returned an empty output tensor");

    const bool classifier = model.model_type == ModelType::Classifier;
    std::vector<Detection> predictions;
    if (classifier) {
        predictions = ClassifierDecoder(model).rank(output.data(), output.size(), model.conf_thresh, 5);
    } else {
        if (model.num_classes < 1) throw std::runtime_error("number of classes must be positive");
        const auto class_count = static_cast<size_t>(model.num_classes);
        const auto values_per_detection = model.version == YOLOVersion::v26 ? 6 :
            class_count + (model.version == YOLOVersion::v5 ? 5 : 4);
        if (output.size() % batch_size != 0 || (output.size() / batch_size) % values_per_detection != 0)
            throw std::runtime_error("model returned an incompatible detection tensor");
        auto decoder = createDecoder(model);
        auto decoded = decoder->decode(output.data(), batch_size, model.input_shape,
                                       model.conf_thresh, model.nms_thresh, output.size());
        if (decoded.empty()) throw std::runtime_error("model returned an invalid detection tensor");
        predictions = std::move(decoded.front());
        mapDetectionsFromModelToFrame(predictions, image.cols, image.rows, model.input_shape);
    }

    auto results = nlohmann::json::array();
    for (const auto& prediction : predictions) {
        if (!std::isfinite(prediction.confidence)) continue;
        const int class_id = prediction.class_id;
        if (class_id < 0 || class_id >= model.num_classes) continue;
        const auto name = static_cast<size_t>(class_id) < model.class_names.size()
            ? model.class_names[class_id] : prediction.class_name;
        nlohmann::json row = {{"class_id", class_id}, {"class_name", name},
                              {"confidence", prediction.confidence}};
        if (!classifier) {
            const auto& box = prediction.bbox;
            if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
                !std::isfinite(box.x2) || !std::isfinite(box.y2)) continue;
            const float x1 = std::clamp(box.x1, 0.f, static_cast<float>(image.cols));
            const float y1 = std::clamp(box.y1, 0.f, static_cast<float>(image.rows));
            const float x2 = std::clamp(box.x2, 0.f, static_cast<float>(image.cols));
            const float y2 = std::clamp(box.y2, 0.f, static_cast<float>(image.rows));
            if (x2 <= x1 || y2 <= y1) continue;
            row["bbox"] = {{"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}};
        }
        results.push_back(std::move(row));
        if (results.size() == kMaxPreviewResults) break;
    }
    auto device = backend.deviceType();
#ifndef BUILD_ONNX_BACKEND_COREML
    if (device == DeviceType::MPS) device = DeviceType::CPU;
#endif
    const bool uses_nms = !classifier && model.version != YOLOVersion::v26;
    return {{"model_type", classifier ? "classifier" : "detector"},
            {"backend", deviceTypeToStr(device)}, {"conf_thresh", model.conf_thresh},
            {"nms_thresh", uses_nms ? nlohmann::json(model.nms_thresh) : nlohmann::json(nullptr)},
            {"width", image.cols}, {"height", image.rows}, {"preview_jpeg", encodePreview(image)},
            {"results", std::move(results)}};
}

} // namespace infer
