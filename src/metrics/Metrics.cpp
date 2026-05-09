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

void Metrics::LabeledCounter::inc(const std::string& label) {
    std::lock_guard lock(mu);
    auto it = data.find(label);
    if (it == data.end()) {
        data.emplace(label, std::make_unique<std::atomic<uint64_t>>(1));
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

void Metrics::LabeledHistogram::observe(const std::string& label, double ms) {
    Histogram* h = nullptr;
    {
        std::lock_guard lock(mu);
        auto it = data.find(label);
        if (it == data.end()) {
            auto inserted = data.emplace(label, std::make_unique<Histogram>());
            h = inserted.first->second.get();
        } else {
            h = it->second.get();
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

void Metrics::recordSinkPublishLatency(const std::string& stream_id, double ms) {
    sink_publish_latency_.observe(stream_id, ms);
}

void Metrics::recordSinkStreamLatency(const std::string& stream_id, double ms) {
    sink_stream_latency_.observe(stream_id, ms);
}

void Metrics::recordInferBatchSize(const std::string& model_id, int size) {
    infer_batch_size_.observe(model_id, static_cast<double>(size));
}

void Metrics::recordInferQueueLatency(const std::string& model_id, double ms) {
    infer_queue_latency_.observe(model_id, ms);
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

// ── Phase 10: stream-level health gauges ──────────────────────────────────────

void Metrics::LabeledGaugeUint::set(const std::string& label, uint64_t v) {
    std::lock_guard lock(mu);
    data[label] = v;
}

void Metrics::LabeledGaugeUint::snapshot(
    std::unordered_map<std::string, uint64_t>& out) const {
    std::lock_guard lock(mu);
    out = data;
}

void Metrics::LabeledGaugeDouble::set(const std::string& label, double v) {
    std::lock_guard lock(mu);
    data[label] = v;
}

void Metrics::LabeledGaugeDouble::snapshot(
    std::unordered_map<std::string, double>& out) const {
    std::lock_guard lock(mu);
    out = data;
}

void Metrics::setStreamState(const std::string& stream_id, uint32_t state) {
    stream_state_.set(stream_id, state);
}

void Metrics::setStreamReconnectCount(const std::string& stream_id, uint32_t n) {
    stream_reconnect_count_.set(stream_id, n);
}

void Metrics::setStreamConsecutiveFailures(const std::string& stream_id, uint32_t n) {
    stream_consecutive_failures_.set(stream_id, n);
}

void Metrics::setStreamLastFrameAgeSeconds(const std::string& stream_id, double age) {
    stream_last_frame_age_.set(stream_id, age);
}

// ── Phase 11: GPU fault self-healing ─────────────────────────────────────────

void Metrics::incGpuOom(const std::string& worker_id) {
    gpu_oom_.inc(worker_id);
}

void Metrics::incGpuEngineFault(const std::string& worker_id) {
    gpu_engine_fault_.inc(worker_id);
}

void Metrics::setInferWorkerState(const std::string& worker_id, uint32_t state) {
    infer_worker_state_.set(worker_id, state);
}

void Metrics::setInferBatchSizeCurrent(const std::string& worker_id, uint32_t size) {
    infer_batch_size_current_.set(worker_id, size);
}

void Metrics::setGpuMemoryUsageRatio(double ratio, const std::string& device_id) {
    gpu_memory_usage_ratio_.set(device_id, ratio);
}

void Metrics::setNpuMemoryUsageRatio(double ratio, const std::string& device_id) {
    npu_memory_usage_ratio_.set(device_id, ratio);
}

// ── SinkFfplayStage metrics ───────────────────────────────────────────────────

void Metrics::recordSinkFfplayJitter(const std::string& stage_id, double ms) {
    sink_jitter_.observe(stage_id, ms);
}

void Metrics::setSinkFfplayQueueDepth(const std::string& stage_id, uint64_t depth) {
    sink_queue_depth_.set(stage_id, depth);
}

void Metrics::incSinkFfplayFramesWritten(const std::string& stage_id) {
    sink_written_.inc(stage_id);
}

void Metrics::incSinkFfplayFramesDropped(const std::string& stage_id) {
    sink_dropped_.inc(stage_id);
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

std::string Metrics::serializeSimpleGauge(
    const std::string& name,
    const std::string& help,
    uint64_t value)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " gauge\n";
    oss << name << " " << value << "\n";
    return oss.str();
}

std::string Metrics::serializeLabeledGaugeUint(
    const std::string& name,
    const std::string& help,
    const std::string& label_name,
    const std::unordered_map<std::string, uint64_t>& data)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " gauge\n";
    for (const auto& [label, val] : data)
        oss << name << "{" << label_name << "=\"" << label << "\"} " << val << "\n";
    return oss.str();
}

std::string Metrics::serializeLabeledGaugeDouble(
    const std::string& name,
    const std::string& help,
    const std::string& label_name,
    const std::unordered_map<std::string, double>& data)
{
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " gauge\n";
    for (const auto& [label, val] : data)
        oss << name << "{" << label_name << "=\"" << label << "\"} "
            << std::fixed << std::setprecision(3) << val << "\n";
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
        oss << std::defaultfloat;  // reset so next label's le= values are unaffected
        oss << name << "_count{" << lbl << "} " << h.count << "\n";
    }
    return oss.str();
}

std::string Metrics::serializeBatchHistogram(
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
        uint64_t cum = 0;
        for (int i = 0; i < kNumBatchBuckets; ++i) {
            cum += h.buckets[i];
            oss << name << "_bucket{" << lbl
                << ",le=\"" << kBatchBounds[i] << "\"} " << cum << "\n";
        }
        oss << name << "_bucket{" << lbl << ",le=\"+Inf\"} " << h.count << "\n";
        oss << name << "_sum{"   << lbl << "} "
            << std::fixed << std::setprecision(3) << h.sum << "\n";
        oss << std::defaultfloat;
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
        std::unordered_map<std::string, HistogramData> snap;
        sink_publish_latency_.snapshot(snap);
        out << serializeHistogram("sink_publish_latency_ms",
            "Latency from frame capture to SinkKafkaStage receive in milliseconds", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, HistogramData> snap;
        sink_stream_latency_.snapshot(snap);
        out << serializeHistogram("sink_stream_latency_ms",
            "Latency from frame capture to DrawAndStreamStage write success in milliseconds", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, HistogramData> snap;
        infer_queue_latency_.snapshot(snap);
        out << serializeHistogram("infer_queue_latency_ms",
            "Time from frame capture to inference start (queue wait + preprocess)", "model_id", snap);
    }
    {
        std::unordered_map<std::string, HistogramData> snap;
        infer_batch_size_.snapshot(snap);
        out << serializeBatchHistogram("infer_batch_size",
            "Distribution of inference batch sizes", "model_id", snap);
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
    out << serializeSimpleGauge("frame_archive_queue_depth",
        "Current pending frame archive queue depth",
        frame_archive_queue_depth_.load(std::memory_order_relaxed));
    {
        std::unordered_map<std::string, uint64_t> snap;
        infer_batches_.snapshot(snap);
        out << serializeCounter("infer_batches_total",
            "Total inference batches processed per model", "model_id", snap);
    }

    // Phase 10: stream-level health gauges
    {
        std::unordered_map<std::string, uint64_t> snap;
        stream_state_.snapshot(snap);
        out << serializeLabeledGaugeUint("stream_state",
            "Current stream state (0=CONNECTING 1=STREAMING 2=RECONNECTING 3=DEGRADED 4=STOPPED)",
            "stream_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        stream_reconnect_count_.snapshot(snap);
        out << serializeLabeledGaugeUint("stream_reconnect_count",
            "Total successful reconnects since start", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        stream_consecutive_failures_.snapshot(snap);
        out << serializeLabeledGaugeUint("stream_consecutive_failures",
            "Consecutive reconnect failures (cleared on success)", "stream_id", snap);
    }
    {
        std::unordered_map<std::string, double> snap;
        stream_last_frame_age_.snapshot(snap);
        out << serializeLabeledGaugeDouble("stream_last_frame_age_seconds",
            "Seconds since last decoded frame arrived", "stream_id", snap);
    }

    // Phase 11: GPU fault self-healing
    {
        std::unordered_map<std::string, uint64_t> snap;
        gpu_oom_.snapshot(snap);
        out << serializeCounter("gpu_oom_total",
            "Cumulative GPU OOM events per worker", "worker_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        gpu_engine_fault_.snapshot(snap);
        out << serializeCounter("gpu_engine_fault_total",
            "Cumulative GPU engine fault events per worker", "worker_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        infer_worker_state_.snapshot(snap);
        out << serializeLabeledGaugeUint("infer_worker_state",
            "InferWorker state (0=RUNNING 1=RECOVERING 2=STOPPED)", "worker_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        infer_batch_size_current_.snapshot(snap);
        out << serializeLabeledGaugeUint("infer_batch_size_current",
            "Current degraded max batch size per worker", "worker_id", snap);
    }
    {
        std::unordered_map<std::string, double> snap;
        gpu_memory_usage_ratio_.snapshot(snap);
        out << serializeLabeledGaugeDouble("gpu_memory_usage_ratio",
            "GPU memory used / total (0-1)", "device", snap);
    }
    {
        std::unordered_map<std::string, double> snap;
        npu_memory_usage_ratio_.snapshot(snap);
        out << serializeLabeledGaugeDouble("npu_memory_usage_ratio",
            "NPU HBM memory used / total (0-1)", "device", snap);
    }

    // SinkFfplayStage metrics
    {
        std::unordered_map<std::string, HistogramData> snap;
        sink_jitter_.snapshot(snap);
        out << serializeHistogram("sink_ffplay_frame_interval_ms",
            "Inter-frame write interval in milliseconds (jitter indicator)", "stage_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        sink_queue_depth_.snapshot(snap);
        out << serializeLabeledGaugeUint("sink_ffplay_queue_depth",
            "Current SinkFfplayStage queue depth", "stage_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        sink_written_.snapshot(snap);
        out << serializeCounter("sink_ffplay_frames_written_total",
            "Total frames successfully written to ffplay", "stage_id", snap);
    }
    {
        std::unordered_map<std::string, uint64_t> snap;
        sink_dropped_.snapshot(snap);
        out << serializeCounter("sink_ffplay_frames_dropped_total",
            "Total frames dropped by SinkFfplayStage queue", "stage_id", snap);
    }

    return out.str();
}

} // namespace infer
