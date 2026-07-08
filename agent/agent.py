from __future__ import annotations

import os
import threading
import time
from typing import Callable, Iterator

import anthropic

from .memory import AgentMemory
from .tasks import Task, TaskQueue, TaskStatus
from .tools import AgentTools

_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 4096
_MAX_ITERATIONS = 12

_SYSTEM = """\
You are GestureAgent, an AI assistant embedded in a hand-gesture control system.
You help the user accomplish goals on their computer by planning subtasks and calling tools.

Capabilities (use them freely):
- Browser: open_url
- Keyboard: type_text, press_key
- Clipboard: read_clipboard, write_clipboard
- Screen/Mouse: get_screen_size, take_screenshot, get_mouse_position, move_mouse, click
- Shell: run_command
- Files: find_files, read_file, write_file
- System: system_info, list_processes, get_active_window

Strategy:
1. Break the goal into concrete subtasks.
2. Execute subtasks with the appropriate tools.
3. If research is needed, use run_command with curl/wget or open_url + read_clipboard.
4. Summarise your result concisely when done.

Past memory (if provided) helps you avoid repeating previous work.
"""


class GestureAgent:
    """AgentGPT-style agentic loop backed by Claude.

    Features:
    - 17 host-machine tools (browser, keyboard, clipboard, mouse, files, system)
    - Persistent JSON memory: recalled automatically per goal
    - Streaming via callback or iterator
    """

    def __init__(
        self,
        api_key: str | None = None,
        memory_path: str | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.queue = TaskQueue()
        self.memory = AgentMemory(memory_path) if memory_path else AgentMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        goal: str,
        *,
        stream_cb: Callable[[str], None] | None = None,
    ) -> Task:
        """Execute *goal* synchronously and return the completed Task."""
        task = self.queue.add(goal)
        task.status = TaskStatus.RUNNING
        try:
            result = self._agent_loop(task, stream_cb=stream_cb)
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as exc:
            task.result = f"Error: {exc}"
            task.status = TaskStatus.FAILED
        finally:
            self.memory.save(task.goal, task.result, task.subtasks)
        return task

    def stream(self, goal: str) -> Iterator[str]:
        """Yield text chunks produced while the agent works toward *goal*.

        Runs the agent in a background thread to avoid blocking the caller.
        """
        chunks: list[str] = []
        lock = threading.Lock()

        def _cb(chunk: str) -> None:
            with lock:
                chunks.append(chunk)

        worker = threading.Thread(
            target=self.run, args=(goal,), kwargs={"stream_cb": _cb}, daemon=True
        )
        worker.start()

        sent = 0
        while worker.is_alive() or sent < len(chunks):
            with lock:
                new_chunks = chunks[sent:]
            for chunk in new_chunks:
                yield chunk
                sent += 1
            if worker.is_alive() and sent == len(chunks):
                time.sleep(0.05)  # yield control; avoid busy-wait

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_system_prompt(self, goal: str) -> str:
        recalled = self.memory.recall(goal, top_k=3)
        if not recalled:
            return _SYSTEM
        mem_lines = "\n".join(
            f"  • [{e.goal[:60]}] → {e.result[:100]}" for e in recalled
        )
        return _SYSTEM + f"\n\nRelevant past tasks:\n{mem_lines}\n"

    def _agent_loop(
        self,
        task: Task,
        *,
        stream_cb: Callable[[str], None] | None = None,
    ) -> str:
        system = self._build_system_prompt(task.goal)
        messages: list[dict] = [{"role": "user", "content": task.goal}]
        tool_schemas = AgentTools.schemas()

        def _emit(text: str) -> None:
            if stream_cb:
                stream_cb(text)

        assistant_content: list[dict] = []

        for _ in range(_MAX_ITERATIONS):
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                system=system,
                tools=tool_schemas,
                thinking={"type": "adaptive"},
                messages=messages,
            )

            tool_calls_made = False
            assistant_content = []
            tool_results: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    _emit(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "thinking":
                    assistant_content.append(
                        {"type": "thinking", "thinking": block.thinking}
                    )
                elif block.type == "tool_use":
                    tool_calls_made = True
                    tool_name = block.name
                    tool_input = block.input or {}
                    task.subtasks.append(f"{tool_name}({tool_input})")
                    _emit(f"\n[{tool_name}] ")
                    result = AgentTools.call(tool_name, **tool_input)
                    _emit(result + "\n")
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn" and not tool_calls_made:
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        final_texts = [b["text"] for b in assistant_content if b.get("type") == "text"]
        return " ".join(final_texts).strip() or "Done."
