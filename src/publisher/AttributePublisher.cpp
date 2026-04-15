#include "publisher/AttributePublisher.h"
#include "common/Logger.h"

namespace infer {

AttributePublisher::AttributePublisher(ResultMerger& merger, std::string attribute_key)
    : merger_(merger)
    , attribute_key_(std::move(attribute_key))
{}

void AttributePublisher::publish(InferResult result) {
    const int det_idx = static_cast<int>(result.frame_seq);

    // Use the top-1 detection's class_name as the attribute value.
    // An empty detections list means the classifier scored below conf_thresh.
    const std::string value = result.detections.empty()
                              ? ""
                              : result.detections[0].class_name;

    merger_.addAttribute(result.stream_id, result.frame_ts,
                         det_idx, attribute_key_, value);
}

} // namespace infer
