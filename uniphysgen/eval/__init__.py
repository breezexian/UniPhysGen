"""Self-contained evaluation for UniPhysGen inference records."""

from .articulation_structure import StructureAccumulator
from .intrinsic_physics_object import ObjectPhysicsAccumulator
from .intrinsic_physics_part import PartPhysicsAccumulator
from .kinematic_parameters import KinematicAccumulator

__all__ = [
    "KinematicAccumulator",
    "ObjectPhysicsAccumulator",
    "PartPhysicsAccumulator",
    "StructureAccumulator",
]
