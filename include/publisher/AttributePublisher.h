#pragma once

#include "publisher/IPublisher.h"
#include "pipeline/ResultMerger.h"
#include <string>

namespace infer {

// Publisher for secondary (classifier) InferWorker instances in a cascade pipeline.
// Instead of sending results to Kafka, it injects the top-1 class label as an
// attribute into the corresponding primary InferResult via ResultMerger.
//
// InferResult fields used:
//   stream_id  → identifies the pending primary result (together with frame_ts)
//   frame_ts   → ditto
//   frame_seq  → det_idx: which detection in the primary result to annotate
//   detections[0].class_name → the attribute value (top-1 label from ClassifierDecoder)
//
// `attribute_key` is the key inserted into Detection.attributes
// (e.g. "vehicle_type", "person_attr").
class AttributePublisher : public IPublisher {
public:
    AttributePublisher(ResultMerger& merger, std::string attribute_key);

    void publish(InferResult result) override;
    void flush() override {}  // no buffering

private:
    ResultMerger& merger_;
    std::string   attribute_key_;
};

} // namespace infer
