#include "decoder/YOLO26Decoder.h"
#include "common/Logger.h"

namespace infer {

// YOLO26 format TBD. Returns empty results until the architecture is published.
std::vector<std::vector<Detection>> YOLO26Decoder::decode(
    const float* /*output*/,
    int          batch_size,
    const InferShape& /*shape*/,
    float        /*conf_thresh*/,
    float        /*nms_thresh*/,
    size_t       /*output_count*/)
{
    LOG_WARN("YOLO26Decoder::decode() not yet implemented — returning empty results");
    return std::vector<std::vector<Detection>>(batch_size);
}

} // namespace infer
