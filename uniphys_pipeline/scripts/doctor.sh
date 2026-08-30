#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${UNIPHYS_PYTHON:-python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/example.yaml}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/pipeline.py" doctor --config "${CONFIG_PATH}"
