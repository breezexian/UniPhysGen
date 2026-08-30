# Copyright (c) PhysLLM Authors.
# All rights reserved.

"""PhysLLM Heads Module.

Scheme B:
- Physics/Group are open-vocabulary generative tasks that use the base LM logits.
    Their "heads" are lightweight prompt/format/parse helpers.
- Motion is a regression head (nn.Module) and is called explicitly in the model.
"""

from .motion_head import MotionHead

__all__ = [
    "MotionHead"
]
