from __future__ import annotations

import fnmatch
import os
import platform
import shutil
import subprocess
import webbrowser
from typing import Any

import pyautogui


# ---------------------------------------------------------------------------
# JSON schemas exposed to Claude — defined first so AgentTools.schemas()
# can reference this at class-definition time.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text using the keyboard.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a key or shortcut (e.g. 'ctrl+c', 'enter', 'alt+tab').",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "read_clipboard",
        "description": "Read the current clipboard contents.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_clipboard",
        "description": "Write text to the clipboard.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "get_screen_size",
        "description": "Return the screen resolution.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "take_screenshot",
        "description": "Capture the screen and save to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Save path (default /tmp/screenshot.png)",
                }
            },
        },
    },
    {
        "name": "get_mouse_position",
        "description": "Return the current mouse cursor position.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "move_mouse",
        "description": "Move the mouse cursor to (x, y).",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "click",
        "description": "Click the mouse at optional (x, y). button: 'left'|'right'|'middle'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
            },
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return output (15 s timeout).",
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "find_files",
        "description": "Find files matching a glob pattern under a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern e.g. '*.py'",
                },
                "directory": {
                    "type": "string",
                    "description": "Root directory to search (default '.')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents (up to 4000 chars).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file (overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "system_info",
        "description": "Return OS, machine type, and free disk space.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_processes",
        "description": "List the top running processes by CPU usage.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_active_window",
        "description": "Return the title of the currently focused window.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class AgentTools:
    """Actions the agent can trigger on the host machine.

    Inspired by SuperAGI's tool registry and gpt-researcher's
    multi-capability approach.
    """

    # ------------------------------------------------------------------ browser
    @staticmethod
    def open_url(url: str) -> str:
        webbrowser.open(url)
        return f"Opened: {url}"

    # ---------------------------------------------------------------- keyboard
    @staticmethod
    def type_text(text: str) -> str:
        pyautogui.typewrite(text, interval=0.03)
        return f"Typed: {text}"

    @staticmethod
    def press_key(key: str) -> str:
        pyautogui.hotkey(*key.split("+"))
        return f"Pressed: {key}"

    # ---------------------------------------------------------------- clipboard
    @staticmethod
    def read_clipboard() -> str:
        try:
            import pyperclip  # type: ignore
            content = pyperclip.paste()
            return content[:2000] if content else "(clipboard empty)"
        except Exception as exc:
            return f"Clipboard error: {exc}"

    @staticmethod
    def write_clipboard(text: str) -> str:
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            return f"Copied to clipboard: {text[:80]}"
        except Exception as exc:
            return f"Clipboard error: {exc}"

    # ------------------------------------------------------------------ screen
    @staticmethod
    def get_screen_size() -> str:
        w, h = pyautogui.size()
        return f"Screen: {w}x{h}"

    @staticmethod
    def take_screenshot(path: str = "/tmp/screenshot.png") -> str:
        try:
            img = pyautogui.screenshot()
            img.save(path)
            return f"Screenshot saved: {path}"
        except Exception as exc:
            return f"Screenshot error: {exc}"

    @staticmethod
    def get_mouse_position() -> str:
        x, y = pyautogui.position()
        return f"Mouse: ({x}, {y})"

    @staticmethod
    def move_mouse(x: int, y: int) -> str:
        pyautogui.moveTo(x, y, duration=0.25)
        return f"Moved mouse to ({x}, {y})"

    @staticmethod
    def click(x: int | None = None, y: int | None = None, button: str = "left") -> str:
        if x is not None and y is not None:
            pyautogui.click(x, y, button=button)
        else:
            pyautogui.click(button=button)
        pos = f"({x}, {y})" if x is not None else "current position"
        return f"Clicked {button} at {pos}"

    # ----------------------------------------------------------------- shell
    @staticmethod
    def run_command(cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=15
            )
            output = result.stdout.strip() or result.stderr.strip() or "Done"
            return output[:3000]
        except subprocess.TimeoutExpired:
            return "Command timed out"

    # ----------------------------------------------------------------- files
    @staticmethod
    def find_files(pattern: str, directory: str = ".") -> str:
        try:
            matches: list[str] = []
            for root, _dirs, files in os.walk(directory):
                for name in fnmatch.filter(files, pattern):
                    matches.append(os.path.join(root, name))
                    if len(matches) >= 50:
                        break
                if len(matches) >= 50:
                    break
            if not matches:
                return f"No files found matching '{pattern}' in '{directory}'"
            return "\n".join(matches)
        except Exception as exc:
            return f"Find error: {exc}"

    @staticmethod
    def read_file(path: str, max_chars: int = 4000) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            return content or "(empty file)"
        except Exception as exc:
            return f"Read error: {exc}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} chars to {path}"
        except Exception as exc:
            return f"Write error: {exc}"

    # ---------------------------------------------------------------- system
    @staticmethod
    def system_info() -> str:
        try:
            # "." works on all platforms (Windows, macOS, Linux)
            disk = shutil.disk_usage(".")
            free_gb = disk.free // (1024 ** 3)
        except Exception:
            free_gb = -1
        return (
            f"OS: {platform.system()} {platform.release()} | "
            f"Machine: {platform.machine()} | "
            f"Disk free: {free_gb} GB"
        )

    @staticmethod
    def list_processes() -> str:
        try:
            if platform.system() == "Windows":
                cmd = ["tasklist"]
            else:
                cmd = ["ps", "aux"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().splitlines()
            return "\n".join(lines[:20])
        except Exception as exc:
            return f"Process list error: {exc}"

    @staticmethod
    def get_active_window() -> str:
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() or "Unknown"
            if platform.system() == "Darwin":
                result = subprocess.run(
                    [
                        "osascript", "-e",
                        "tell application \"System Events\" to get name of "
                        "first application process whose frontmost is true",
                    ],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip() or "Unknown"
            return "get_active_window not supported on this OS"
        except Exception as exc:
            return f"Window error: {exc}"

    # ---------------------------------------------------------------- registry
    REGISTRY: dict[str, Any] = {}

    @classmethod
    def _build_registry(cls) -> None:
        cls.REGISTRY = {
            "open_url":           cls.open_url,
            "type_text":          cls.type_text,
            "press_key":          cls.press_key,
            "read_clipboard":     cls.read_clipboard,
            "write_clipboard":    cls.write_clipboard,
            "get_screen_size":    cls.get_screen_size,
            "take_screenshot":    cls.take_screenshot,
            "get_mouse_position": cls.get_mouse_position,
            "move_mouse":         cls.move_mouse,
            "click":              cls.click,
            "run_command":        cls.run_command,
            "find_files":         cls.find_files,
            "read_file":          cls.read_file,
            "write_file":         cls.write_file,
            "system_info":        cls.system_info,
            "list_processes":     cls.list_processes,
            "get_active_window":  cls.get_active_window,
        }

    @classmethod
    def call(cls, name: str, **kwargs: Any) -> str:
        if not cls.REGISTRY:
            cls._build_registry()
        fn = cls.REGISTRY.get(name)
        if fn is None:
            return f"Unknown tool: {name}"
        return fn(**kwargs)

    @classmethod
    def schemas(cls) -> list[dict]:
        return TOOL_SCHEMAS


AgentTools._build_registry()
