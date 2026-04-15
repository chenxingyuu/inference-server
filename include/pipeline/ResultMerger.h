#pragma once

#include "common/Types.h"
#include "publisher/IPublisher.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <functional>

namespace infer {

// Merges primary InferResults with secondary classifier results.
//
// Flow:
//   1. CascadeRouter registers a primary result via registerPrimary()
//   2. Secondary InferWorker calls addAttribute() for each ROI classified
//   3. Once all expected attributes arrive (or deadline passes), flush() publishes
//
// Thread-safety: registerPrimary() and addAttribute() may be called from
// different threads. Both are guarded by a mutex.
class ResultMerger {
public:
    // `max_wait_ms`: how long to wait for secondary results before publishing
    // the primary with whatever attributes have arrived.
    explicit ResultMerger(IPublisher& publisher, int max_wait_ms = 20);

    // Register a primary result that expects `expected_count` secondary attribute injections.
    // If expected_count == 0, the result is published immediately.
    void registerPrimary(InferResult      result,
                         int              expected_count);

    // Called by secondary InferWorker: inject an attribute into a pending result.
    // `stream_id` + `frame_ts` identifies the primary result.
    // `det_idx` is the detection index within InferResult.detections.
    // `attribute_key` / `attribute_value` are merged into Detection.attributes.
    void addAttribute(const std::string& stream_id,
                      double             frame_ts,
                      int                det_idx,
                      const std::string& attribute_key,
                      const std::string& attribute_value);

    // Flush all entries whose deadline has passed. Should be called periodically
    // (e.g. by a background thread or by the InferWorker after each batch).
    void flushExpired();

private:
    struct PendingEntry {
        InferResult result;
        int         expected;
        int         received{0};
        std::chrono::steady_clock::time_point deadline;
    };

    void publishAndErase(std::unordered_map<std::string, PendingEntry>::iterator it);

    IPublisher& publisher_;
    int         max_wait_ms_;

    std::mutex                                    mu_;
    std::unordered_map<std::string, PendingEntry> pending_;  // key = stream_id:frame_seq
};

} // namespace infer
