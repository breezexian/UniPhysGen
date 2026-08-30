# Copyright (c) PhysLLM Authors.
# All rights reserved.

"""Optional motion-regression heads for UniPhysGen.

UniPhysGenQwen3ForCausalLM calls MotionHead for the motion task when
use_motion_head is enabled. Generative tasks use the shared language-model head.
"""

from .motion_head import MotionHead

__all__ = [
    "MotionHead"
]
