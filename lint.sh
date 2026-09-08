#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -d '' PYTHON_FILES < <(find "$ROOT_DIR" -type f -name "*.py" -print0)

if [ "${#PYTHON_FILES[@]}" -eq 0 ]; then
  echo "No Python files found."
  exit 0
fi

ruff check --fix "${PYTHON_FILES[@]}"
ruff format "${PYTHON_FILES[@]}"

echo "Linting applied to ${#PYTHON_FILES[@]} Python file(s)."
