from __future__ import annotations

from uniphys.cli import _parse_blender_version


def test_parse_blender_version() -> None:
    output = "Blender 4.5.3\n\tbuild date: 2025-10-07"

    assert _parse_blender_version(output) == (4, 5, 3)


def test_parse_blender_version_rejects_unknown_output() -> None:
    assert _parse_blender_version("not Blender output") is None
