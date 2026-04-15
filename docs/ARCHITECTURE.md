# Architecture

`inference-server` is a C++ real-time video inference service for multi-stream RTSP ingest and low-latency object detection.

## Runtime Pipeline

1. Stream decode (`FFmpegDecoder` / optional HW decode).
2. Buffer and schedule by model (`FrameBuffer`, `BatchScheduler`).
3. Backend inference (`TRTBackend` or `AscendBackend`).
4. YOLO decode and postprocess.
5. Async publish to Kafka and expose management endpoints.

## Architecture Boundaries

- `include/` and `src/`: service runtime and backend implementations.
- `config/`: runtime configuration and observability manifests.
- `docker/`: build/runtime containerization.
- `scripts/`: operational utilities.
- `docs/`: durable engineering guidance.

## Non-Goals

- No ad-hoc runtime behavior hidden only in chat history.
- No policy that cannot be enforced through scripts, CI, or tests.
