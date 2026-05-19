# Product Rules

Durable product-level constraints for `inference-server`.

## Core Invariants

- The service supports dynamic **task** management: the C++ engine exposes NDJSON commands on a **Unix domain socket** (`server.socket_path`, default `/var/run/infer.sock`); the Go **`infer-server`** sidecar forwards HTTP (`/healthz`, `/metrics`, `/tasks`, model repository APIs, etc.) to that socket. **`infer-ctl`** is the CLI for the same protocol.
- Inference behavior is model-scoped and driven by `pipelines[].nodes` referencing `models[].id`; `tasks[]` selects which graph runs for which source.
- Telemetry endpoints exposed by **`infer-server`** (`/healthz`, `/metrics`, `/tasks`, …) remain available to operators and Prometheus.
- Stream lifecycle semantics are externally observable:
  - Heartbeat continues to emit per-stream `stream_state`
  - Control events are emitted to the dedicated Kafka control topic (default `inference-control`)
- Runtime changes must preserve existing backend split (`TensorRT` vs `Ascend`).

## Change Discipline

- Any behavior change affecting API, config, or runtime semantics must update docs.
- Large changes should be split into small verifiable iterations.
- Completion claims must include verification evidence and residual risk.

## Documentation Impact

When code paths change, update at least one relevant document:

- `docs/ARCHITECTURE.md` for architecture boundary changes
- `README.md` for usage/operations changes
- `docs/iterations.md` for notable implementation milestones
- `docs/ascend-guide.md` for Ascend/DVPP/AIPP/VPC behavior or conversion workflow changes
