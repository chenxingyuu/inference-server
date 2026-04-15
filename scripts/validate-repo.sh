#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

use_rg=0
if command -v rg >/dev/null 2>&1; then
  use_rg=1
else
  echo "WARN: ripgrep (rg) not found; falling back to grep -E."
fi

match_lines() {
  local pattern="$1"
  if [[ "${use_rg}" -eq 1 ]]; then
    rg "${pattern}" || true
  else
    grep -E "${pattern}" || true
  fi
}

DOCS_GATE_MODE="${DOCS_GATE_MODE:-warn}"
CODE_PATHS_REGEX="${CODE_PATHS_REGEX:-^(src/|include/|tests/|cmake/|config/|docker/|CMakeLists\\.txt$|build_tests/)}"

required_files=(
  "AGENTS.md"
  "docs/INDEX.md"
  "docs/ARCHITECTURE.md"
  "docs/PRODUCT_RULES.md"
  "docs/QUALITY-GATES.md"
)

missing=0
echo "== Baseline required files =="
for f in "${required_files[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "MISSING: ${f}"
    missing=1
  else
    echo "OK: ${f}"
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "FAIL: missing required baseline files."
  exit 1
fi

BASE_REF="${BASE_REF:-origin/main}"
if git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
  diff_range="${BASE_REF}...HEAD"
else
  diff_range="HEAD~1...HEAD"
fi

changed_files="$(git diff --name-only "${diff_range}" || true)"
if [[ -z "${changed_files}" ]]; then
  changed_files="$(git diff --name-only || true)"
fi

echo
echo "== Changed files (${diff_range}) =="
if [[ -n "${changed_files}" ]]; then
  echo "${changed_files}"
else
  echo "(none)"
fi

code_files="$(printf '%s\n' "${changed_files}" | match_lines "${CODE_PATHS_REGEX}")"
doc_files="$(printf '%s\n' "${changed_files}" | match_lines '^(docs/|README\.md$|AGENTS\.md$)')"

docs_skip_reason="${DOCS_SKIP_REASON:-}"
if [[ -z "${docs_skip_reason}" ]]; then
  combined_text="$(printf '%s\n%s\n' "${PR_BODY:-}" "${COMMIT_MESSAGE:-}")"
  docs_skip_reason="$(printf '%s' "${combined_text}" | grep -Eo '\[docs-skip:[^]]+\]' | head -n1 || true)"
fi

echo
echo "== Docs-sync gate =="
echo "Mode: ${DOCS_GATE_MODE}"
echo "Code-trigger files:"
if [[ -n "${code_files}" ]]; then
  echo "${code_files}"
else
  echo "(none)"
fi
echo "Docs files changed:"
if [[ -n "${doc_files}" ]]; then
  echo "${doc_files}"
else
  echo "(none)"
fi
echo "Docs skip reason:"
if [[ -n "${docs_skip_reason}" ]]; then
  echo "${docs_skip_reason}"
else
  echo "(none)"
fi

if [[ -n "${code_files}" && -z "${doc_files}" && -z "${docs_skip_reason}" ]]; then
  echo
  echo "Docs-sync gate not satisfied."
  echo "Remediation:"
  echo "  1) Update docs in docs/**, README.md, or AGENTS.md"
  echo "  2) OR add explicit reason: [docs-skip: <reason>] or DOCS_SKIP_REASON"
  if [[ "${DOCS_GATE_MODE}" == "enforce" ]]; then
    echo "FAIL: docs-sync gate enforced."
    exit 1
  fi
  echo "WARN: docs-sync gate warning only."
fi

echo
echo "PASS: validate-repo checks completed."
