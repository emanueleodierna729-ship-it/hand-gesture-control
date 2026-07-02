from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class Task:
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    subtasks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "result": self.result,
            "subtasks": self.subtasks,
            "created_at": self.created_at,
        }


class TaskQueue:
    def __init__(self) -> None:
        self._tasks: list[Task] = []

    def add(self, goal: str) -> Task:
        task = Task(goal=goal)
        self._tasks.append(task)
        return task

    def pending(self) -> list[Task]:
        return [t for t in self._tasks if t.status == TaskStatus.PENDING]

    def completed(self) -> list[Task]:
        return [t for t in self._tasks if t.status == TaskStatus.COMPLETED]

    def all(self) -> list[Task]:
        return list(self._tasks)

    def clear(self) -> None:
        self._tasks.clear()
