"""Task-specific prompt/label helpers.

These helpers are intentionally lightweight: they only define prompt skeletons
(with placeholders) and JSON label formatting.

They are used by dataset converters/formatters to build ShareGPT messages.
"""

from .registry import TASK_REGISTRY, get_task_handler

__all__ = [
	"TASK_REGISTRY",
	"get_task_handler"
]
