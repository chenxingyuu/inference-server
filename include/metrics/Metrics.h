#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace infer {

// Lightweight Prometheus-compatible metrics exporter.
// No external library dependency — implements the text exposition format
// directly using atomic counters and simple histogram buckets.
//
// Thread-safe: counters use std::atomic; histograms use a per-histogram mutex.
class Metrics {
public:
    static Metrics& get();

    // ── Latency histograms (milliseconds) ─────────────────────────────────────
    void recordInferLatency(const std::string& model_id, double ms);
    void recordE2eLatency(const std::string& stream_id, double ms);

    // ── Counters ──────────────────────────────────────────────────────────────
    void incFramesDecoded(const std::string& stream_id);
    void incFramesDropped(const std::string& stream_id);
    void incKafkaPublished();
    void incKafkaDropped();
    void incInferBatches(const std::string& model_id);
    void incFramesArchived();
    void incFramesArchiveDropped();
    void incFramesUploaded();
    void incFramesUploadFailed();
    void setFrameArchiveQueueDepth(uint64_t depth);

    // ── Exposition ────────────────────────────────────────────────────────────
    // Returns the full Prometheus text format (OpenMetrics compatible).
    std::string serialize() const;

    static constexpr double kHistBounds[] = {
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
    };
    static constexpr int kNumHistBuckets = 10;

private:
    Metrics() = default;

    // Plain-old-data snapshot (no mutex, copyable)
    struct HistogramData {
        uint64_t buckets[kNumHistBuckets]{};
        uint64_t count{0};
        double   sum{0.0};
    };

    // Live histogram entry (has mutex, non-copyable)
    struct Histogram {
        mutable std::mutex mu;
        uint64_t           buckets[kNumHistBuckets]{};
        uint64_t           count{0};
        double             sum{0.0};

        void observe(double ms);
    };

    // Labeled counter map: label_value → count
    struct LabeledCounter {
        mutable std::mutex                                       mu;
        std::unordered_map<std::string, std::atomic<uint64_t>*> data;

        ~LabeledCounter();
        void inc(const std::string& label);
        void snapshot(std::unordered_map<std::string, uint64_t>& out) const;
    };

    // Labeled histogram map: label_value → Histogram
    struct LabeledHistogram {
        mutable std::mutex                           mu;
        std::unordered_map<std::string, Histogram*> data;

        ~LabeledHistogram();
        void observe(const std::string& label, double ms);
        void snapshot(std::unordered_map<std::string, HistogramData>& out) const;
    };

    LabeledHistogram      infer_latency_;
    LabeledHistogram      e2e_latency_;
    LabeledCounter        frames_decoded_;
    LabeledCounter        frames_dropped_;
    std::atomic<uint64_t> kafka_published_{0};
    std::atomic<uint64_t> kafka_dropped_{0};
    LabeledCounter        infer_batches_;
    std::atomic<uint64_t> frames_archived_{0};
    std::atomic<uint64_t> frames_archive_dropped_{0};
    std::atomic<uint64_t> frames_uploaded_{0};
    std::atomic<uint64_t> frames_upload_failed_{0};
    std::atomic<uint64_t> frame_archive_queue_depth_{0};

    // Serialization helpers
    static std::string serializeCounter(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, uint64_t>& data);

    static std::string serializeSimpleCounter(
        const std::string& name,
        const std::string& help,
        uint64_t value);
    static std::string serializeSimpleGauge(
        const std::string& name,
        const std::string& help,
        uint64_t value);

    static std::string serializeHistogram(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, HistogramData>& data);
};

} // namespace infer
