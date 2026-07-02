from __future__ import annotations

import threading
from typing import Callable

from .agent import GestureAgent
from .tasks import Task

_TRIGGER_GESTURES = {
    "AGENT_OPEN":   "apri browser",
    "AGENT_COPY":   "copia testo selezionato",
    "AGENT_PASTE":  "incolla testo",
    "AGENT_SAVE":   "salva file corrente",
    "AGENT_UNDO":   "annulla ultima azione",
    "AGENT_SCREEN": "cattura screenshot",
}


class GestureBridge:
    """Maps named gesture events to GestureAgent goals and runs them asynchronously."""

    def __init__(
        self,
        agent: GestureAgent | None = None,
        on_start: Callable[[str, str], None] | None = None,
        on_done: Callable[[Task], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> None:
        self.agent = agent or GestureAgent()
        self._on_start = on_start
        self._on_done = on_done
        self._on_chunk = on_chunk
        self._active: dict[str, threading.Thread] = {}

    def trigger(self, gesture_name: str, custom_goal: str | None = None) -> bool:
        """Fire the agent for *gesture_name*. Returns False if already running."""
        goal = custom_goal or _TRIGGER_GESTURES.get(gesture_name)
        if not goal:
            return False
        if self._active.get(gesture_name, threading.Thread()).is_alive():
            return False

        if self._on_start:
            self._on_start(gesture_name, goal)

        def _run() -> None:
            task = self.agent.run(goal, stream_cb=self._on_chunk)
            if self._on_done:
                self._on_done(task)

        t = threading.Thread(target=_run, daemon=True, name=f"agent-{gesture_name}")
        self._active[gesture_name] = t
        t.start()
        return True

    def is_busy(self) -> bool:
        return any(t.is_alive() for t in self._active.values())

    @property
    def trigger_map(self) -> dict[str, str]:
        return dict(_TRIGGER_GESTURES)
