# Architecture

`inference-server` is a C++ real-time video inference service for multi-stream RTSP ingest and low-latency object detection.

## Runtime Pipeline

The runtime is a **configurable in-process DAG** driven by **tasks** (each task binds one `source_id` to one reusable `pipeline_id` template). Per-task **`sample_fps`** and **`use_hwdec`** control RTSP ingest rate and FFmpeg hardware decode for that task’s `source.rtsp` instance; **`sources`** entries carry `url` and reconnect policy only (deprecated `sources[].sample_fps` / `sources[].use_hwdec` keys are rejected at load time).

1. Source ingest (`source.rtsp` using `FFmpegDecoder`; optional HW decode via NVDEC).
2. Fan-out to parallel branches (e.g. `archive.raw` and inference path).
3. Optional frame archiving (`archive.raw` via `FrameArchiver` → local JPEG + MinIO).
4. Inference (`infer.engine` → `InferEngineStage` using `TRTBackend` / `AscendBackend` / `OnnxBackend`) with per-edge backpressure. `InferEngineStage` accumulates frames into a batch and flushes when the batch is full **or** a background deadline timer fires (`max_queue_delay_us / 2` poll interval), ensuring low-fps streams are not stalled waiting for the next frame.
5. YOLO decode (`IYOLODecoder` / `ClassifierDecoder`) and optional tracking (`track.bytetrack`).
6. Optional join/merge (`join.byFrameId`) to enrich inference results with archive metadata.
7. Optional draw + output: restream via ffmpeg pipe (`sink.stream` → RTSP/RTMP), or local preview via `ffplay` stdin (`sink.ffplay`, raw BGR).
8. Publish via `buildPublisher()` factory (see **Publishers** section below).
9. Background heartbeat emission (`HeartbeatPublisher`) and management endpoints (`ManagementServer`).

## Publishers

Result publishing is configurable via the `publishers:` YAML block. All publishers implement `IPublisher` and are wired by `buildPublisher()` in `main.cpp`.

| Publisher | Enabled by default | CMake flag | Notes |
|-----------|-------------------|------------|-------|
| `KafkaPublisher` | yes | always built | async, batched; Kafka topic per config |
| `RedisPublisher` | no | `BUILD_REDIS_PUBLISHER=ON` | async XADD to Redis Streams; stream key `{prefix}:{stream_id}` |
| `GrpcPublisher` | no | `BUILD_GRPC_PUBLISHER=ON` | server-streaming RPC (`InferenceService::Subscribe`); slow subscribers dropped at 64 pending frames |
| `MultiPublisher` | — | always built | fan-out adapter; used automatically when ≥2 publishers are enabled |

When only one publisher is enabled, `buildPublisher()` returns it directly with no wrapping overhead. Legacy YAML configs with a root-level `kafka:` key automatically map to `publishers.kafka` for backward compat.

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
- **Management API** (`GET /healthz`, `GET /tasks`): liveness probe and task-level start/stop (`POST /tasks/{id}/start|stop`).
- **Structured logs** (spdlog, `[stream_id] state: OLD -> NEW` format):
  - `StreamHealthRegistry` emits `INFO`/`WARN`/`ERROR` on every state transition so stream-level events are always visible at the default `info` log level.
  - `KafkaPublisher` / `RedisPublisher` log `ERROR` when the outbound queue is full and results are dropped.
  - `ResultMerger` logs `WARN` when a cascade secondary deadline expires and an incomplete result is published.
  - `initLogger()` configures `flush_on(error)` so ERROR-level records are flushed to disk immediately, surviving process crashes.

## Architecture Boundaries

- `include/` and `src/`: service runtime and backend implementations.
- Pipeline stage implementations live under `include/pipeline/stages/` and `src/pipeline/stages/`; `StageFactory` wires YAML `type` strings to those classes.
- `config/`: runtime configuration and observability manifests.
- `docker/`: build/runtime containerization.
- `scripts/`: operational utilities.
- `docs/`: durable engineering guidance.

## Non-Goals

- No ad-hoc runtime behavior hidden only in chat history.
- No policy that cannot be enforced through scripts, CI, or tests.
