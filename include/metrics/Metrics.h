#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <memory>

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

    // ── Batch size histogram ───────────────────────────────────────────────────
    void recordInferBatchSize(const std::string& model_id, int size);

    // ── Infer queue wait latency histogram ────────────────────────────────────
    // Time from frame capture to batch inference start (queue wait + preprocess).
    void recordInferQueueLatency(const std::string& model_id, double ms);

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

    // ── SinkFfplayStage metrics ───────────────────────────────────────────────
    void recordSinkFfplayJitter(const std::string& stage_id, double ms);
    void setSinkFfplayQueueDepth(const std::string& stage_id, uint64_t depth);
    void incSinkFfplayFramesWritten(const std::string& stage_id);
    void incSinkFfplayFramesDropped(const std::string& stage_id);

    // ── Phase 10: stream-level health gauges ──────────────────────────────────
    // Updated by HeartbeatPublisher on each heartbeat interval (not on hot path).
    void setStreamState(const std::string& stream_id, uint32_t state);
    void setStreamReconnectCount(const std::string& stream_id, uint32_t n);
    void setStreamConsecutiveFailures(const std::string& stream_id, uint32_t n);
    void setStreamLastFrameAgeSeconds(const std::string& stream_id, double age);

    // ── Phase 11: GPU fault self-healing ──────────────────────────────────────
    void incGpuOom(const std::string& worker_id);
    void incGpuEngineFault(const std::string& worker_id);
    void setInferWorkerState(const std::string& worker_id, uint32_t state);
    void setInferBatchSizeCurrent(const std::string& worker_id, uint32_t size);
    // device_id: e.g. "0" for GPU 0. ratio in [0, 1].
    void setGpuMemoryUsageRatio(double ratio, const std::string& device_id = "0");
    // device_id: e.g. "0" for NPU 0. ratio in [0, 1].
    void setNpuMemoryUsageRatio(double ratio, const std::string& device_id = "0");

    // ── Exposition ────────────────────────────────────────────────────────────
    // Returns the full Prometheus text format (OpenMetrics compatible).
    std::string serialize() const;

    static constexpr double kHistBounds[] = {
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
    };
    static constexpr int kNumHistBuckets = 10;

    // Batch-size histogram uses power-of-two bounds (1..32)
    static constexpr double kBatchBounds[] = { 1, 2, 4, 8, 16, 32 };
    static constexpr int kNumBatchBuckets = 6;

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
        std::unordered_map<std::string, std::unique_ptr<std::atomic<uint64_t>>> data;

        void inc(const std::string& label);
        void snapshot(std::unordered_map<std::string, uint64_t>& out) const;
    };

    // Labeled gauge map: label_value → uint64 (state, counts)
    struct LabeledGaugeUint {
        mutable std::mutex                         mu;
        std::unordered_map<std::string, uint64_t> data;

        void set(const std::string& label, uint64_t v);
        void snapshot(std::unordered_map<std::string, uint64_t>& out) const;
    };

    // Labeled gauge map: label_value → double (age in seconds, etc.)
    struct LabeledGaugeDouble {
        mutable std::mutex                       mu;
        std::unordered_map<std::string, double> data;

        void set(const std::string& label, double v);
        void snapshot(std::unordered_map<std::string, double>& out) const;
    };

    // Labeled histogram map: label_value → Histogram
    struct LabeledHistogram {
        mutable std::mutex                           mu;
        std::unordered_map<std::string, std::unique_ptr<Histogram>> data;

        void observe(const std::string& label, double ms);
        void snapshot(std::unordered_map<std::string, HistogramData>& out) const;
    };

    LabeledHistogram      infer_latency_;
    LabeledHistogram      e2e_latency_;
    LabeledHistogram      infer_batch_size_;
    LabeledHistogram      infer_queue_latency_;
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

    // Phase 10: stream-level health gauges
    LabeledGaugeUint   stream_state_;
    LabeledGaugeUint   stream_reconnect_count_;
    LabeledGaugeUint   stream_consecutive_failures_;
    LabeledGaugeDouble stream_last_frame_age_;

    // Phase 11: GPU fault self-healing
    LabeledCounter     gpu_oom_;
    LabeledCounter     gpu_engine_fault_;
    LabeledGaugeUint   infer_worker_state_;
    LabeledGaugeUint   infer_batch_size_current_;
    LabeledGaugeDouble gpu_memory_usage_ratio_; // label_name="device", value in [0,1]
    LabeledGaugeDouble npu_memory_usage_ratio_; // label_name="device", value in [0,1]

    // SinkFfplayStage
    LabeledHistogram sink_jitter_;
    LabeledGaugeUint sink_queue_depth_;
    LabeledCounter   sink_written_;
    LabeledCounter   sink_dropped_;

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

    static std::string serializeLabeledGaugeUint(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, uint64_t>& data);

    static std::string serializeLabeledGaugeDouble(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, double>& data);

    static std::string serializeHistogram(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, HistogramData>& data);

    // Variant that uses kBatchBounds/kNumBatchBuckets instead of kHistBounds
    static std::string serializeBatchHistogram(
        const std::string& name,
        const std::string& help,
        const std::string& label_name,
        const std::unordered_map<std::string, HistogramData>& data);
};

} // namespace infer
