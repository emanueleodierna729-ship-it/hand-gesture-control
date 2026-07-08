from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_DEFAULT_PATH = os.path.expanduser("~/.gesture_agent_memory.json")
_MAX_ENTRIES = 200


@dataclass
class MemoryEntry:
    goal: str
    result: str
    subtasks: list[str]
    timestamp: float


class AgentMemory:
    """Simple JSON-backed persistent memory inspired by gpt-researcher's
    research history and SuperAGI's resource manager."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._entries: list[MemoryEntry] = []
        self._load()

    # ---------------------------------------------------------------- public
    def save(self, goal: str, result: str, subtasks: list[str] | None = None) -> None:
        entry = MemoryEntry(
            goal=goal,
            result=result,
            subtasks=subtasks or [],
            timestamp=time.time(),
        )
        self._entries.append(entry)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._persist()

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Return the most recent entries whose goal contains query keywords."""
        keywords = query.lower().split()
        scored: list[tuple[int, MemoryEntry]] = []
        for e in self._entries:
            score = sum(k in e.goal.lower() or k in e.result.lower() for k in keywords)
            if score:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [e for _, e in scored[:top_k]]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        return list(reversed(self._entries[-n:]))

    def summary(self) -> str:
        if not self._entries:
            return "No memory yet."
        lines = [f"- [{e.goal[:60]}] → {e.result[:80]}" for e in self.recent(5)]
        return "Recent tasks:\n" + "\n".join(lines)

    def clear(self) -> None:
        self._entries.clear()
        self._persist()

    # --------------------------------------------------------------- private
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            if not isinstance(raw, list):
                self._entries = []
                return
            self._entries = [
                MemoryEntry(**r) for r in raw
                if isinstance(r, dict) and all(
                    k in r for k in ("goal", "result", "subtasks", "timestamp")
                )
            ]
        except Exception:
            self._entries = []

    def _persist(self) -> None:
        try:
            self._path.write_text(
                json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2)
            )
        except Exception:
            pass
