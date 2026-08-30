#!/usr/bin/env python3
"""Public command-line entry point for the UniPhys pipeline.

The orchestrator lives in the :mod:`uniphys` package and is re-exported here
so repository-based workflows have a short, stable command.
"""

from uniphys.cli import main
from uniphys.core import PipelineRunner, Stage, StageRegistry

__all__ = ["PipelineRunner", "Stage", "StageRegistry", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
