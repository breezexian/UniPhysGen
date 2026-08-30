from __future__ import annotations

from typing import Dict, Type
from .motion import MotionTask
from .physics import PhysicsTask
from .group import GroupTask
from .object_level import ObjectLevelTask

from .base import BaseTaskHandler


TaskHandler = Type[BaseTaskHandler]


TASK_REGISTRY: Dict[str, TaskHandler] = {
    MotionTask.TASK_NAME: MotionTask,
    PhysicsTask.TASK_NAME: PhysicsTask,
    GroupTask.TASK_NAME: GroupTask,
    ObjectLevelTask.TASK_NAME: ObjectLevelTask,
}


def get_task_handler(task_name: str) -> TaskHandler:
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_name={task_name!r}. Available: {sorted(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name]
