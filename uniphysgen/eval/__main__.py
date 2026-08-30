"""Unified entry point: python -m eval TASK INPUT [INPUT ...]."""

from __future__ import annotations

import sys

from .articulation_structure import main as structure_main
from .intrinsic_physics_object import main as object_main
from .intrinsic_physics_part import main as part_main
from .kinematic_parameters import main as kinematic_main


COMMANDS = {
    "physics": part_main,
    "intrinsic_physics_part": part_main,
    "part": part_main,
    "object_level": object_main,
    "intrinsic_physics_object": object_main,
    "object": object_main,
    "motion": kinematic_main,
    "kinematic_parameters": kinematic_main,
    "kinematic": kinematic_main,
    "group": structure_main,
    "articulation_structure": structure_main,
    "structure": structure_main,
}


def main() -> None:
    choices = "|".join(sorted(COMMANDS))
    usage = f"usage: python -m eval {{{choices}}} INPUT [INPUT ...]"
    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        print(usage)
        return
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(usage)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
