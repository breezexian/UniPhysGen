#!/usr/bin/env python3
"""Read-only checks for common public-release blockers.

The audit intentionally reports file locations without printing credential
values. It does not rewrite Git history or remove local artifacts.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 50 * 1024 * 1024
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
)
FORBIDDEN_TRACKED_PARTS = {".DS_Store", "__pycache__"}
FORBIDDEN_TRACKED_PREFIXES = (".idea/", "outputs/", "exp_results/")
FORBIDDEN_TRACKED_SUFFIXES = {".ckpt", ".pth", ".pt", ".pyc", ".so"}
TEXT_SUFFIXES = {
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token)\s*[=:]\s*['\"][^'\"]{12,}['\"]"),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [PROJECT_ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    blockers: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative).is_file():
            blockers.append(f"missing required file: {relative}")

    try:
        files = tracked_files()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: cannot inspect tracked files: {exc}", file=sys.stderr)
        return 2

    for path in files:
        # Ignore tracked files already removed from the working tree. They will
        # disappear from the index when the release changes are staged.
        if not path.exists():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if path.name in FORBIDDEN_TRACKED_PARTS or relative.startswith(
            FORBIDDEN_TRACKED_PREFIXES
        ):
            blockers.append(f"generated/editor artifact is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES:
            blockers.append(f"binary/model artifact is tracked: {relative}")
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if size > MAX_TRACKED_BYTES:
            blockers.append(
                f"large tracked file ({size / 1024 / 1024:.1f} MiB): {relative}"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES or size > 5 * 1024 * 1024:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            warnings.append(f"cannot scan {relative}: {exc}")
            continue
        for line_number, line in enumerate(lines, 1):
            if "your_api_key" in line or "sk-xxx" in line:
                continue
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                blockers.append(f"possible credential: {relative}:{line_number}")

    sam_license = PROJECT_ROOT / "utils/sam2/LICENSE"
    if not sam_license.is_file():
        warnings.append(
            "SAM2 license/attribution file is not present at utils/sam2/LICENSE"
        )

    if blockers:
        print("Release blockers:")
        for blocker in sorted(set(blockers)):
            print(f"  - {blocker}")
    if warnings:
        print("Warnings:")
        for warning in sorted(set(warnings)):
            print(f"  - {warning}")
    if not blockers:
        print("No automated release blockers found.")
    print(f"Audit summary: blockers={len(set(blockers))} warnings={len(set(warnings))}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
