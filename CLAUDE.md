# CLAUDE.md

Project-level instructions for Claude Code in this repository.

## Documentation

When updating or creating documentation (iteration docs, project summaries), keep them concise and match the existing style. Do not make docs verbose unless explicitly asked.

## Development Workflow

Follow TDD (RED-GREEN-REFACTOR) workflow for all C++ and Python feature implementations unless told otherwise.
Do not create git commits automatically after modifying code. Only commit when the user explicitly asks for a commit.
When committing, include only changes made for the current request. Do not stage or commit unrelated work from the user or other sessions.

## Implementation Checklist

When implementing features, ensure ALL integration points are covered (API endpoints, config entries, CMake targets, docs) before reporting completion. Do not skip peripheral files.

## Debugging

When instrumenting or debugging code, verify you are modifying the ACTIVE code path (production pipeline), not dead code or debug-only tools like ffplay.
