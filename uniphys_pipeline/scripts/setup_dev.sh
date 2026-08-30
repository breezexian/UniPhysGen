#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="${UNIPHYS_CONDA_ENV:-uniphys_pipeline}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required but was not found on PATH." >&2
  exit 1
fi

if conda run -n "${CONDA_ENV_NAME}" python --version >/dev/null 2>&1; then
  if ! conda run -n "${CONDA_ENV_NAME}" python -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    echo "Conda environment ${CONDA_ENV_NAME} exists but does not use Python 3.10." >&2
    echo "Choose another name with UNIPHYS_CONDA_ENV or recreate the environment." >&2
    exit 1
  fi
else
  conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

conda run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${CONDA_ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}[runtime,dev]"

echo "Development environment ready: ${CONDA_ENV_NAME}"
echo "Activate it with: conda activate ${CONDA_ENV_NAME}"
echo "Then run: scripts/doctor.sh configs/local.yaml"
