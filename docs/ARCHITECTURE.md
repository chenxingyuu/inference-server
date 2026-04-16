# Architecture

`inference-server` is a C++ real-time video inference service for multi-stream RTSP ingest and low-latency object detection.

## Runtime Pipeline

The runtime is a **configurable in-process DAG pipeline**:

1. Source ingest (`source.rtsp` using `FFmpegDecoder`; optional HW decode via NVDEC).
2. Fan-out to parallel branches (e.g. `archive.raw` and inference path).
3. Optional frame archiving (`archive.raw` via `FrameArchiver` → local JPEG + MinIO).
4. Inference (`infer.engine` using `TRTBackend` / `AscendBackend` / `OnnxBackend`) with per-edge backpressure.
5. YOLO decode (`IYOLODecoder` / `ClassifierDecoder`) and optional tracking (`track.bytetrack`).
6. Optional join/merge (`join.byFrameId`) to enrich inference results with archive metadata.
7. Publish (`sink.kafka` via `KafkaPublisher`).
8. Background heartbeat emission (`HeartbeatPublisher`) and management endpoints (`ManagementServer`).

## Key Singletons

| Singleton | Role |
|-----------|------|
| `Metrics` | Prometheus-format counters and histograms; written from hot paths via atomics |
| `StreamHealthRegistry` | Per-stream state machine (`CONNECTING → STREAMING → RECONNECTING → DEGRADED → FAILED → STOPPED`); written by `FFmpegDecoder`, read by `HeartbeatPublisher` and `ManagementServer` |
| `ControlEventBus` | Best-effort stream control event emission (wired to `ControlPublisher` in `main`) |

## Observability

- **Metrics** (`GET /metrics`): Prometheus text format scraped by Prometheus every 15 s.
- **Heartbeat** (`inference-heartbeat` Kafka topic): per-stream and engine-level heartbeat every 5 s. Downstream can distinguish: `STREAMING`+no-frames=no targets; `RECONNECTING/DEGRADED`=camera issue; heartbeat stops=engine down.
- **Control events** (`inference-control` Kafka topic): stream lifecycle events (`stream_dropped`, `stream_recovered`, `stream_failed_terminal`) emitted by `FFmpegDecoder`.
- **Management API** (`GET /healthz`, `GET /pipelines`): liveness probe and pipeline-level operations.

## Architecture Boundaries

- `include/` and `src/`: service runtime and backend implementations.
- `config/`: runtime configuration and observability manifests.
- `docker/`: build/runtime containerization.
- `scripts/`: operational utilities.
- `docs/`: durable engineering guidance.

## Non-Goals

- No ad-hoc runtime behavior hidden only in chat history.
- No policy that cannot be enforced through scripts, CI, or tests.
