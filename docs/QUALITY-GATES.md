# Quality Gates

This repository uses a single executable quality gate: `scripts/validate-repo.sh`.

## Required Baseline Files

The validation script fails if any required harness files are missing:

- `AGENTS.md`
- `docs/INDEX.md`
- `docs/ARCHITECTURE.md`
- `docs/PRODUCT_RULES.md`
- `docs/QUALITY-GATES.md`

## Docs-Sync Gate

When code files change, docs must be updated in the same change set unless an explicit skip reason is provided.

### Triggered Code Paths

`CODE_PATHS_REGEX` default:

`^(src/|include/|tests/|cmake/|config/|docker/|CMakeLists\\.txt$|build_tests/)`

### Accepted Docs Paths

- `docs/**`
- `README.md`
- `AGENTS.md`

### Skip Marker

Use one of these explicit skip channels:

- commit or PR text contains `[docs-skip: <reason>]`
- `DOCS_SKIP_REASON` environment variable is set

### Gate Modes

- `DOCS_GATE_MODE=warn` (default): warning only
- `DOCS_GATE_MODE=enforce`: hard failure when docs-sync is missing

## Completion Rule

Before claiming completion for substantial work:

```bash
scripts/validate-repo.sh
```
