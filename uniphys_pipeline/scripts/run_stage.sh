#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_stage.sh STAGE [CONFIG] [pipeline arguments...]" >&2
  exit 64
fi

STAGE="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${UNIPHYS_PYTHON:-python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/example.yaml}"

if [[ $# -gt 0 ]]; then
  shift
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/pipeline.py" run \
  --config "${CONFIG_PATH}" \
  --stages "${STAGE}" \
  "$@"
