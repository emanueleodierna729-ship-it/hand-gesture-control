from __future__ import annotations

import os
from typing import Iterator

import anthropic

from .tasks import Task, TaskQueue, TaskStatus
from .tools import AgentTools

_SYSTEM = """\
You are GestureAgent, an AI assistant embedded in a hand-gesture control system.
You help the user accomplish goals on their computer by planning subtasks and calling tools.
Available tools: open_url, type_text, press_key, run_command, get_screen_size, system_info.
When given a goal, think step-by-step and call tools as needed. Summarise your result concisely.
"""

_TOOL_SCHEMAS = [
    {
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to open"}},
            "required": ["url"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text using the keyboard.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to type"}},
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a key or keyboard shortcut (e.g. 'ctrl+c', 'enter').",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key or combo to press"}},
            "required": ["key"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its output (10 s timeout).",
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "Shell command to execute"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "get_screen_size",
        "description": "Return the current screen resolution.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "system_info",
        "description": "Return OS and machine information.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 4096
_MAX_ITERATIONS = 10


class GestureAgent:
    """AgentGPT-style agentic loop backed by Claude."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.queue = TaskQueue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, goal: str, *, stream_cb: "((str) -> None) | None" = None) -> Task:
        """Execute *goal* and return the completed Task."""
        task = self.queue.add(goal)
        task.status = TaskStatus.RUNNING
        try:
            result = self._agent_loop(task, stream_cb=stream_cb)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as exc:
            task.result = f"Error: {exc}"
            task.status = TaskStatus.FAILED
        return task

    def stream(self, goal: str) -> Iterator[str]:
        """Yield text chunks as the agent works toward *goal*."""
        chunks: list[str] = []

        def _cb(chunk: str) -> None:
            chunks.append(chunk)

        import threading
        task_holder: list[Task] = []

        def _worker() -> None:
            task_holder.append(self.run(goal, stream_cb=_cb))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        sent = 0
        while thread.is_alive() or sent < len(chunks):
            while sent < len(chunks):
                yield chunks[sent]
                sent += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _agent_loop(self, task: Task, *, stream_cb: "((str) -> None) | None" = None) -> str:
        messages: list[dict] = [{"role": "user", "content": task.goal}]

        def _emit(text: str) -> None:
            if stream_cb:
                stream_cb(text)

        for _ in range(_MAX_ITERATIONS):
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM,
                tools=_TOOL_SCHEMAS,
                thinking={"type": "adaptive"},
                messages=messages,
            )

            tool_calls_made = False
            assistant_content: list[dict] = []
            tool_results: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    _emit(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "thinking":
                    assistant_content.append({"type": "thinking", "thinking": block.thinking})
                elif block.type == "tool_use":
                    tool_calls_made = True
                    tool_name = block.name
                    tool_input = block.input or {}
                    task.subtasks.append(f"Tool: {tool_name}({tool_input})")
                    _emit(f"\n[{tool_name}] ")
                    result = AgentTools.call(tool_name, **tool_input)
                    _emit(result + "\n")
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": tool_name,
                        "input": tool_input,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn" and not tool_calls_made:
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        final_texts = [
            b["text"] for b in assistant_content if b.get("type") == "text"
        ]
        return " ".join(final_texts).strip() or "Done."
