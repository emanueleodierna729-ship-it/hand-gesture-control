from __future__ import annotations

import subprocess
import webbrowser
import platform
import pyautogui
from typing import Any


class AgentTools:
    """Actions the agent can trigger on the host machine."""

    @staticmethod
    def open_url(url: str) -> str:
        webbrowser.open(url)
        return f"Opened: {url}"

    @staticmethod
    def type_text(text: str) -> str:
        pyautogui.typewrite(text, interval=0.03)
        return f"Typed: {text}"

    @staticmethod
    def press_key(key: str) -> str:
        pyautogui.hotkey(*key.split("+"))
        return f"Pressed: {key}"

    @staticmethod
    def run_command(cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() or result.stderr.strip() or "Done"
        except subprocess.TimeoutExpired:
            return "Command timed out"

    @staticmethod
    def get_screen_size() -> str:
        w, h = pyautogui.size()
        return f"Screen: {w}x{h}"

    @staticmethod
    def system_info() -> str:
        return (
            f"OS: {platform.system()} {platform.release()} | "
            f"Machine: {platform.machine()}"
        )

    REGISTRY: dict[str, Any] = {}

    @classmethod
    def _build_registry(cls) -> None:
        cls.REGISTRY = {
            "open_url":     cls.open_url,
            "type_text":    cls.type_text,
            "press_key":    cls.press_key,
            "run_command":  cls.run_command,
            "get_screen_size": cls.get_screen_size,
            "system_info":  cls.system_info,
        }

    @classmethod
    def call(cls, name: str, **kwargs: Any) -> str:
        if not cls.REGISTRY:
            cls._build_registry()
        fn = cls.REGISTRY.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        return fn(**kwargs)


AgentTools._build_registry()
