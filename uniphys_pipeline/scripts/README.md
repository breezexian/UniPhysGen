# Helper scripts

These scripts are intentionally thin wrappers. Pipeline selection, state,
dependency checks, error handling, and parallelism remain in Python.

- `run_pipeline.sh [CONFIG] [ARGS...]` runs the configured stage sequence.
- `run_stage.sh STAGE [CONFIG] [ARGS...]` runs one named or numbered stage.
- `doctor.sh [CONFIG]` checks executables, interpreters, checkpoints, prompts,
  and required environment variables.
- `setup_dev.sh` creates the Python 3.10 Conda environment
  `uniphys_pipeline` and installs runtime and development tooling. Set
  `UNIPHYS_CONDA_ENV` to choose another name. SAM2 and PartField remain in
  their official, separate Conda environments.
- `release_audit.py` performs read-only checks for credentials, large tracked
  files, editor artifacts, required metadata, and missing third-party notices.

The wrappers use `python` by default. Set `UNIPHYS_PYTHON` when the
orchestration interpreter has a different path.
