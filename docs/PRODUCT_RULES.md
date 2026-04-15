# Product Rules

Durable product-level constraints for `inference-server`.

## Core Invariants

- The service supports dynamic stream management through HTTP endpoints.
- Inference behavior is model-scoped and batch-scheduled by `model_id`.
- Telemetry endpoints (`/healthz`, `/metrics`, `/streams`) remain available.
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
