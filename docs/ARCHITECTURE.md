# Architecture

`inference-server` is a C++ real-time video inference service for multi-stream RTSP ingest and low-latency object detection.

## Runtime Pipeline

1. Stream decode (`FFmpegDecoder` / optional HW decode via NVDEC).
2. Buffer and schedule by model (`FrameBuffer`, `BatchScheduler`).
3. Backend inference (`TRTBackend` or `AscendBackend`).
4. YOLO decode and postprocess (`IYOLODecoder` / `ClassifierDecoder`).
5. Optional cascade: `CascadeRouter` → secondary classifier → `ResultMerger`.
6. Optional per-stream tracking (`none` / `bytetrack`, `deepsort` placeholder).
7. Optional frame archiving (`FrameArchiver` → local JPEG + MinIO).
8. Async publish to Kafka (`KafkaPublisher`).
9. Background heartbeat emission (`HeartbeatPublisher`) and management endpoints (`ManagementServer`).

## Key Singletons

| Singleton | Role |
|-----------|------|
| `Metrics` | Prometheus-format counters and histograms; written from hot paths via atomics |
| `StreamHealthRegistry` | Per-stream state machine (`CONNECTING → STREAMING → RECONNECTING → DEGRADED → STOPPED`); written by `FFmpegDecoder`, read by `HeartbeatPublisher` and `ManagementServer` |

## Observability

- **Metrics** (`GET /metrics`): Prometheus text format scraped by Prometheus every 15 s.
- **Heartbeat** (`inference-heartbeat` Kafka topic): per-stream and engine-level heartbeat every 5 s. Downstream can distinguish: `STREAMING`+no-frames=no targets; `RECONNECTING/DEGRADED`=camera issue; heartbeat stops=engine down.
- **Management API** (`GET /health`, `GET /streams/{id}/health`): readiness probe and per-stream detail.

## Architecture Boundaries

- `include/` and `src/`: service runtime and backend implementations.
- `config/`: runtime configuration and observability manifests.
- `docker/`: build/runtime containerization.
- `scripts/`: operational utilities.
- `docs/`: durable engineering guidance.

## Non-Goals

- No ad-hoc runtime behavior hidden only in chat history.
- No policy that cannot be enforced through scripts, CI, or tests.
