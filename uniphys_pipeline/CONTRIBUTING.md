# Contributing

Thank you for improving UniPhys Pipeline.

## Development setup

1. Fork and clone the repository.
2. Run `scripts/setup_dev.sh` or create an equivalent Python 3.10+ environment.
3. Run `python pipeline.py list-stages` and `python -m pytest`.
4. Install optional GPU environments only for the stages being changed.

## Change guidelines

- Keep orchestration logic in `uniphys/`; keep shell scripts thin.
- Do not add machine-specific absolute paths, credentials, checkpoints, output
  directories, compiled extensions, or generated media.
- Preserve stage input/output contracts and update output validation when a
  contract changes.
- Add CPU-only tests for configuration, selection, state, and error semantics.
- For algorithm changes, document the expected metric or output difference and
  provide a reproducible configuration.
- Prefer argument-list subprocess calls with `check=True`; do not use
  `shell=True` for paths derived from users or datasets.

## Pull requests

A pull request should contain one coherent change, a clear motivation, test
instructions, and documentation updates. Confirm that:

```bash
python -m pytest
python -m ruff check pipeline.py uniphys tests
python -m mypy pipeline.py uniphys
python pipeline.py list-stages
```

all succeed before requesting review.

## Third-party contributions

Do not copy code, models, or assets without compatible redistribution terms.
Record new third-party components in `THIRD_PARTY_NOTICES.md` and preserve the
upstream copyright and license text.
