# Architecture

`inference-server` is a C++ real-time video inference service for multi-stream RTSP ingest and low-latency object detection.

## Runtime Pipeline

The runtime is a **configurable in-process DAG** driven by **tasks** (each task binds one `source_id` to one reusable `pipeline_id` template). Per-task **`sample_fps`**, **`sampling_mode`**, and **`use_hwdec`** control RTSP ingest rate, sampling strategy, and FFmpeg hardware decode for that task’s `source.rtsp` / `source.file` instance; **`sources`** entries carry `url` and reconnect policy only (deprecated `sources[].sample_fps` / `sources[].use_hwdec` keys are rejected at load time).

Two sampling modes are available via `tasks[].sampling_mode`:
- **`frame_count`** (default): emit every `N`th decoded frame where `N = round(actual_stream_fps / sample_fps)`, computed from `avg_frame_rate` after the stream opens.
- **`time_based`**: emit a frame only when `frame_pts_sec − last_emit_pts ≥ 1.0 / sample_fps`; falls back to `frame_count` logic when PTS is unavailable.

1. Source ingest (`source.rtsp` / `source.file`): default `FFmpegDecoder` (optional NVDEC via `use_hwdec`); on Ascend builds with `use_ascend_dvpp: true`, `DVPPDecoder` demuxes with FFmpeg and hardware-decodes to NPU HBM (see **Ascend ingest** below).
2. Fan-out to parallel branches (e.g. `archive.raw` and inference path).
3. Optional frame archiving (`archive.raw` via `FrameArchiver` → local JPEG).
4. Inference (`infer.engine` → `InferEngineWorkerStage` → `InferWorkerGroup` using `TRTBackend` / `AscendBackend` / `OnnxBackend`) with per-edge backpressure. The stage accumulates frames into a batch and flushes when the batch is full **or** a background deadline timer fires (`max_queue_delay_us / 2` poll interval), ensuring low-fps streams are not stalled waiting for the next frame. `models[].instance_count` and `models[].device_ids` select parallel workers (see `InferWorkerGroup`). A separate hot-path (`ModelManager` + `BatchScheduler` + `InferWorkerGroup`) still exists for stream-pool–centric scheduling; the DAG stage does not use `BatchScheduler`.
5. YOLO decode (`IYOLODecoder` / `ClassifierDecoder`) and optional tracking (`track.bytetrack`) — decode runs inside each `InferWorker` after the backend forward pass.
6. Optional join/merge (`join.byFrameId`) to enrich inference results with archive metadata.
7. Optional draw + output: restream via ffmpeg pipe (`sink.stream` → RTSP/RTMP), or local preview via `ffplay` stdin (`sink.ffplay`, raw BGR). For task-shared pipelines, `sink.stream.with.output_url` supports task-time placeholders `{task_id}` / `{source_id}` (unknown tokens are rejected).
8. Publish via `buildPublisher()` factory (see **Publishers** section below).
9. Background heartbeat emission (`HeartbeatPublisher`) and the management plane (see **Management plane** below).

## Management plane

Since Phase 26, the C++ process does **not** embed an HTTP server. Management uses a **Unix domain socket** plus a Go sidecar:

```text
inferenced (C++)  --NDJSON on server.socket_path-->  infer-server (Go) --HTTP--> operators / Prometheus / infer-web
                                                   infer-ctl (Go CLI) --same socket-->
```

- **`UnixSocketServer`** (`src/server/UnixSocketServer.cpp`): newline-delimited JSON request/response (`list_tasks`, `start_task`, `stop_task`, `update_task`, `update_pipeline`, repository load, etc.).
- **`tools/infer-server`**: HTTP reverse proxy to the socket (`GET /healthz`, `GET /metrics`, `GET/POST /tasks`, model repository REST, Swagger).
- **`tools/infer-ctl`**: CLI for the same commands; `INFER_SOCKET` overrides the socket path.

Legacy references to `ManagementServer` (cpp-httplib) are removed.

## Ascend ingest (DVPP + optional VPC)

When `BUILD_ASCEND_BACKEND` is enabled and a task sets `use_ascend_dvpp: true`:

1. **`SourceRtspStage`** constructs `DVPPDecoder` with `ascend_device_id` and optional VPC output size.
2. **`TaskManager`** sets `StreamConfig.ascend_vpc_out_width/height` from the pipeline’s `infer.engine` model `input_shape` (0 = no VPC). If codec resolution already matches the model input, VPC is skipped.
3. **`DVPPDecoder`**: VDEC → optional **`DvppVpcScaler`** (YUV420SP resize on HBM) → `Frame.is_ascend` / `ascend_buf` into the DAG.
4. **`InferEngineWorkerStage`** batches into `Batch.ascend_frames`; **`AscendBackend`** Path A (`is_ascend && aipp_enabled_`) passes device pointers to the embedded AIPP in `.om` (`scripts/aipp.cfg` typically `src=640×640`).

Operational detail: `docs/ascend-guide.md` §12–§14.

## Model registry (YAML + optional repository)

- Root `config.yaml` may declare models under `models:` as today.
- If `server.model_repository` is set (non-empty), or `INFER_MODEL_REPOSITORY` is set in the environment (the **env var overrides** the YAML value after parse), `loadConfig()` scans that directory and **appends** discovered models into `AppConfig::models` before validation.
- Layout per model: `<repo>/<model_id>/config.yaml` plus numeric version directories `<repo>/<model_id>/<n>/` holding weights. Optional keys in the per-model `config.yaml`: `active_version` (int), `weight_file` (string). Relative `engine_path` / `onnx_path` / `om_paths` values resolve against the selected version directory.
- Duplicate `id` between root `models:` and repository models is a **hard error**. `cascade[].model_id` references must resolve after this merge (`validateCascadeModelRefs`).
- **Hot-swap caveat**: DAG stages snapshot `ModelConfig` at graph build time (`StageFactory` → `InferEngineWorkerStage`); changing files on disk does not update an already-built graph without a process restart or explicit rebuild of executors.

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
| `StreamHealthRegistry` | Per-stream state machine (`CONNECTING → STREAMING → RECONNECTING → DEGRADED → FAILED → STOPPED`); written by decoders (`FFmpegDecoder`, `DVPPDecoder`), read by `HeartbeatPublisher` and management queries via `TaskManager` |
| `ControlEventBus` | Best-effort stream control event emission (wired to `ControlPublisher` in `main`) |

## Observability

- **Metrics** (`GET /metrics`): Prometheus text format scraped by Prometheus every 15 s.
- **Heartbeat** (`inference-heartbeat` Kafka topic): per-stream and engine-level heartbeat every 5 s. Downstream can distinguish: `STREAMING`+no-frames=no targets; `RECONNECTING/DEGRADED`=camera issue; heartbeat stops=engine down.
- **Control events** (`inference-control` Kafka topic): stream lifecycle events (`stream_dropped`, `stream_recovered`, `stream_failed_terminal`) emitted by `FFmpegDecoder`.
- **Management API** (HTTP on **`infer-server`**, not in-process C++): `GET /healthz` (liveness), `GET /tasks`, `POST /tasks/{id}/start|stop`, pipeline/task updates, model repository APIs — all forwarded to `UnixSocketServer` on `server.socket_path`.
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
