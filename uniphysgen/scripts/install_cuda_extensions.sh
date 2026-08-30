#!/usr/bin/env bash
set -Eeuo pipefail

# These packages require custom build or wheel-index options that cannot be
# represented portably by standard project dependencies in pyproject.toml.
python -m pip install ninja
python -m pip install --no-cache-dir flash-attn --no-build-isolation
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install spconv-cu120
