#include "metrics/Metrics.h"
#include <sstream>
#include <iomanip>

namespace infer {

// ── Histogram::observe ────────────────────────────────────────────────────────

void Metrics::Histogram::observe(double ms) {
    std::lock_guard lock(mu);
    count++;
    sum += ms;
    // Increment only the first bucket whose upper bound >= ms.
    // The serializer accumulates these into Prometheus cumulative bucket counts.
    for (int i = 0; i < kNumHistBuckets; ++i) {
        if (ms <= kHistBounds[i]) {
            buckets[i]++;
            return;
        }
    }
    // Exceeds all bounds — `count` acts as the +Inf bucket.
}

// ── LabeledCounter ────────────────────────────────────────────────────────────

Metrics::LabeledCounter::~LabeledCounter() {
    for (auto& [k, v] : data) delete v;
}

void Metrics::LabeledCounter::inc(const std::string& label) {
    std::lock_guard lock(mu);
    auto it = data.find(label);
    if (it == data.end()) {
        data.emplace(label, new std::atomic<uint64_t>(1));
    } else {
        it->second->fetch_add(1, std::memory_order_relaxed);
    }
}

void Metrics::LabeledCounter::snapshot(
    std::unordered_map<std::string, uint64_t>& out) const {
    std::lock_guard lock(mu);
    for (const auto& [k, v] : data) {
        out[k] = v->load(std::memory_order_relaxed);
    }
}

// ── LabeledHistogram ──────────────────────────────────────────────────────────

Metrics::LabeledHistogram::~LabeledHistogram() {
    for (auto& [k, v] : data) delete v;
}

void Metrics::LabeledHistogram::observe(const std::string& label, double ms) {
    Histogram* h = nullptr;
    {
        std::lock_guard lock(mu);
        auto it = data.find(label);
        if (it == data.end()) {
            h = new Histogram{};
            data.emplace(label, h);
        } else {
            h = it->second;
        }
    }
    h->observe(ms);
}

void Metrics::LabeledHistogram::snapshot(
    std::unordered_map<std::string, HistogramData>& out) const {
    std::lock_guard lock(mu);
    for (const auto& [k, v] : data) {
        HistogramData snap{};
        std::lock_guard hlock(v->mu);
        snap.count = v->count;
        snap.sum   = v->sum;
        for (int i = 0; i < kNumHistBuckets; ++i)
            snap.buckets[i] = v->buckets[i];
        out.emplace(k, snap);
    }
}

// ── Metrics singleton ─────────────────────────────────────────────────────────

Metrics& Metrics::get() {
    static Metrics instance;
    return instance;
}

void Metrics::recordInferLatency(const std::string& model_id, double ms) {
    infer_latency_.observe(model_id, ms);
}

void Metrics::recordE2eLatency(const std::string& stream_id, double ms) {
    e2e_latency_.observe(stream_id, ms);
}

void Metrics::incFramesDecoded(const std::string& stream_id) {
    frames_decoded_.inc(stream_id);
}

void Metrics::incFramesDropped(const std::string& stream_id) {
    frames_dropped_.inc(stream_id);
}

void Metrics::incKafkaPublished() {
    kafka_published_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::incKafkaDropped() {
    kafka_dropped_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::incInferBatches(const std::string& model_id) {
    infer_batches_.inc(model_id);
}

void Metrics::incFramesArchived() {
    frames_archived_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::incFramesArchiveDropped() {
    frames_archive_dropped_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::incFramesUploaded() {
    frames_uploaded_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::incFramesUploadFailed() {
    frames_upload_failed_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::setFrameArchiveQueueDepth(uint64_t depth) {
    frame_archive_queue_depth_.store(depth, std::memory_order_relaxed);
}

// ── Serialization ─────────────────────────────────────────────────────────────

std::string Metrics::serializeCounter(
    const std::string& name,
    const std::string& help,
    const std::string& label_name,
    const std::unordered_map<std::string, uint64_t>& data)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " counter\n";
    for (const auto& [label, val] : data) {
        oss << name << "{" << label_name << "=\"" << label << "\"} " << val << "\n";
    }
    return oss.str();
}

std::string Metrics::serializeSimpleCounter(
    const std::string& name,
    const std::string& help,
    uint64_t value)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " counter\n";
    oss << name << " " << value << "\n";
    return oss.str();
}

std::string Metrics::serializeHistogram(
    const std::string& name,
    const std::string& help,
    const std::string& label_name,
    const std::unordered_map<std::string, HistogramData>& data)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " histogram\n";
    for (const auto& [label, h] : data) {
        const std::string lbl = label_name + "=\"" + label + "\"";
        // Cumulative buckets
        uint64_t cum = 0;
        for (int i = 0; i < kNumHistBuckets; ++i) {
            cum += h.buckets[i];
            oss << name << "_bucket{" << lbl
                << ",le=\"" << kHistBounds[i] << "\"} " << cum << "\n";
        }
        oss << name << "_bucket{" << lbl << ",le=\"+Inf\"} " << h.count << "\n";
        oss << name << "_sum{"   << lbl << "} "
            << std::fixed << std::setprecision(3) << h.sum << "\n";
        oss << name << "_count{" << lbl << "} " << h.count << "\n";
    }
    return oss.str();
}

std::string Metrics::serialize() const {
    std::ostringstream out;

    {
        std::unordered_map<std::string, HistogramData> snap;
        infer_latency_.snapshot(snap);
        out << serializeHistogram("infer_latency_ms",
            "Inference latency per batch in milliseconds", "model_id", snap);
    }
    {
        std::unordered_map<std::string, HistogramData> snap;
        e2e_latency_.snapshot(snap);
        out << serializeHistogram("e2e_latency_ms",
            "End-to-end latency (capture to publish) in milliseconds", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        frames_decoded_.snapshot(snap);
        out << serializeCounter("frames_decoded_total",
            "Total frames decoded per stream", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        frames_dropped_.snapshot(snap);
        out << serializeCounter("frames_dropped_total",
            "Total frames dropped due to full frame buffer per stream", "stream_id", snap);
    }
    out << serializeSimpleCounter("kafka_published_total",
        "Total messages successfully produced to Kafka",
        kafka_published_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("kafka_dropped_total",
        "Total messages dropped due to full publish queue",
        kafka_dropped_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("frames_archived_total",
        "Total frames archived to local storage",
        frames_archived_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("frames_archive_dropped_total",
        "Total frames dropped by archive queue or write failures",
        frames_archive_dropped_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("frames_uploaded_total",
        "Total frames uploaded to MinIO",
        frames_uploaded_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("frames_upload_failed_total",
        "Total frame upload failures",
        frames_upload_failed_.load(std::memory_order_relaxed));
    out << serializeSimpleCounter("frame_archive_queue_depth",
        "Current pending frame archive queue depth",
        frame_archive_queue_depth_.load(std::memory_order_relaxed));
    {
        std::unordered_map<std::string, uint64_t> snap;
        infer_batches_.snapshot(snap);
        out << serializeCounter("infer_batches_total",
            "Total inference batches processed per model", "model_id", snap);
    }

    return out.str();
}

} // namespace infer
