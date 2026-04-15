# AGENTS

Execution entrypoint for agent-driven work in `inference-server`.

## 1) Start Here

- Read `docs/INDEX.md` first for navigation.
- Read `docs/QUALITY-GATES.md` before making changes.
- Run `scripts/validate-repo.sh` before claiming completion.

## 2) Working Contract

- Keep durable constraints in versioned files, not chat memory.
- Prefer small, verifiable diffs over large unverified batches.
- Reuse existing architecture and naming patterns from current code.
- Include evidence for validation commands in completion reports.

## 3) Required Completion Output

For substantial tasks, include:

- **Scope**: files changed and intent
- **Verification**: commands run and outcomes
- **Risk**: known gaps or unverified paths
- **Next Action**: smallest high-value follow-up

## 4) Fast Validation

```bash
scripts/validate-repo.sh
```
