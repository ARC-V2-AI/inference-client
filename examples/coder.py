"""Textual coding-agent UI for the ARC V2 inference client.

Run from this SDK repository:
    uv sync --extra tui
    uv run example.py --workspace ~/Projects/my-project

The main agent owns its transcript locally. It can delegate one bounded task to
an ephemeral subagent. Both use the same inference client and workspace tools;
the runtime serializes inference and clears shared KV contents after each
request. Subagent history is discarded when delegation returns.

Workspace writes and shell commands require interactive approval. Shell
commands run with the current user's permissions and are NOT sandboxed merely
because their working directory is the workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    Switch,
    TextArea,
)

from arc_v2_inference_client import AgentEvent, InferenceClient, StreamEvent, tool

DEFAULT_URL = os.getenv("ARC_RUNTIME_URL", "http://127.0.0.1:7842")
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}
MAX_READ_CHARS = 16_000
MAX_TOOL_OUTPUT = 8_000
DEFAULT_SUBAGENT_CONTEXT = 8_192


def _env_context_limit() -> int:
    """Read the requested context default; runtime status remains authoritative."""
    try:
        return max(256, int(os.getenv("ARC_RUNTIME_N_CTX", "16384")))
    except (TypeError, ValueError):
        return 16_384


MAIN_INSTRUCTIONS = """You are ARC, a local-first coding agent working inside the user's selected workspace.

Your job is to inspect, implement, test and explain software changes. Be precise and
prefer small, reviewable changes. Inspect relevant files before editing. Use the
workspace tools rather than inventing file contents or claiming work you did not do.

TOOLS AND SAFETY
- list_files, search_text and read_file are read-only.
- write_file changes a workspace file and always requires user approval.
- run_command requires the user to enable shell commands and approve every command.
  Commands run with the user's operating-system permissions. The workspace is only
  the working directory; it is NOT a security sandbox.
- Never attempt to bypass a denied or rejected operation. Continue without it or
  explain the blocker.
- Do not read credentials, private keys, browser profiles, or unrelated personal files.
- Do not delete files. Do not run destructive commands unless the user explicitly
  requests the exact operation and separately approves the command.

SUBAGENTS
- You can delegate one self-contained implementation, inspection, or review task to
  delegate_to_subagent. Use it when parallel expertise or an isolated task is useful,
  not for trivial work.
- The subagent has its own temporary local history and the same workspace tools,
  but cannot create further subagents.
- The subagent returns a result to you; incorporate and verify that result yourself.
- Do not claim the subagent changed or tested something unless its returned tool
  results support that claim.

WORKFLOW
1. Inspect the workspace and identify relevant files.
2. State a concise plan before substantial edits.
3. Make focused changes. Ask for approval through tools when required.
4. Run relevant checks when permitted.
5. Summarize files changed, checks performed, and remaining limitations.

Use subagents as often as possible to lay of tasks and thus preserver your own precious context window.
"""

SUBAGENT_INSTRUCTIONS = """You are a temporary ARC coding subagent.

Complete only the delegated task. Work within the selected workspace. Inspect files
before editing. Use read-only tools freely. Every write and shell command requires
user approval. Shell commands are NOT sandboxed and run with the user's permissions.
Do not delete files, do not delegate further, and do not retry denied operations.
Return a concise report containing: findings, files changed, checks run, and blockers.
Do not claim success without tool evidence.
"""


class ApprovalScreen(ModalScreen[bool]):
    """Modal approval for a single write or shell command."""

    BINDINGS = [("escape", "deny", "Deny")]
    CSS = """
    ApprovalScreen { align: center middle; background: $background 80%; }
    #approval-box { width: 88; max-width: 95%; height: auto; max-height: 85%;
                    padding: 1 2; border: thick $warning; background: $surface; }
    #approval-title { text-style: bold; color: $warning; margin-bottom: 1; }
    #approval-detail { height: auto; max-height: 20; border: round $panel;
                       padding: 1; overflow-y: auto; }
    #approval-warning { color: $error; margin-top: 1; }
    #approval-buttons { height: 3; margin-top: 1; align-horizontal: right; }
    #approval-buttons Button { margin-left: 1; }
    """

    def __init__(self, title: str, detail: str, warning: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.detail_text = detail
        self.warning_text = warning

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(self.title_text, id="approval-title")
            yield Static(self.detail_text, id="approval-detail", markup=False)
            if self.warning_text:
                yield Static(self.warning_text, id="approval-warning", markup=False)
            with Horizontal(id="approval-buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Approve once", id="approve", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_deny(self) -> None:
        self.dismiss(False)


class CodingAgentApp(App[None]):
    TITLE = "ARC V2 · Coding Agent"
    SUB_TITLE = "Local inference · client-owned agent state · temporary subagents"
    # Priority bindings are evaluated ahead of focused-widget bindings. This is
    # important because TextArea handles several Ctrl-key combinations itself.
    BINDINGS = [
        Binding("f2", "send_prompt", "Send", show=True, priority=True),
        Binding("ctrl+enter", "send_prompt", "Send", show=False, priority=True),
        Binding("ctrl+shift+enter", "send_prompt", "Send", show=False, priority=True),
        Binding("f6", "clear_history", "Clear history", show=True, priority=True),
        Binding("ctrl+l", "clear_history", "Clear history", show=False, priority=True),
        Binding(
            "ctrl+shift+l", "clear_history", "Clear history", show=False, priority=True
        ),
        Binding("f5", "refresh_runtime", "Refresh runtime", show=True, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=True, priority=True),
    ]

    CSS = """
    Screen { background: #0b1020; color: #dce5f2; }
    #workspace { height: 1fr; padding: 0 1 1 1; }
    #main-pane { width: 3fr; min-width: 52; border: round #334155; padding: 1; }
    #side-pane { width: 48; min-width: 40; border: round #334155; padding: 1; margin-left: 1; height: 1fr; overflow-y: auto; }
    .section-title { color: #7dd3fc; text-style: bold; height: 1; margin-bottom: 1; }
    #conversation { height: 1fr; border: round #263449; padding: 0 1; scrollbar-gutter: stable; }
    #live-output { height: 10; min-height: 5; border: round #2563eb; padding: 1; overflow-y: auto; }
    #composer { height: 10; margin-top: 1; }
    #prompt { height: 6; border: round #475569; }
    #buttons { height: 3; margin-top: 1; }
    #buttons Button { margin-right: 1; }
    #token-panel, #agent-panel, #subagent-panel, #runtime-panel, #settings-panel {
        border: round #263449; padding: 1; margin-bottom: 1; height: auto;
    }
    .muted { color: #94a3b8; }
    .metric { color: #e2e8f0; }
    .user-line { color: #93c5fd; text-style: bold; }
    .assistant-line { color: #86efac; text-style: bold; }
    .tool-line { color: #fcd34d; }
    .error-line { color: #fca5a5; }
    .setting-label { color: #94a3b8; margin-top: 1; }
    .setting-input { height: 3; }
    #settings-row { height: 3; }
    #settings-row Input { width: 1fr; margin-right: 1; }
    #shell-row { height: 3; }
    #reasoning-row { height: 3; }
    #runtime-panel { height: auto; min-height: 7; }
    #runtime-details { height: auto; min-height: 5; }
    #memory-panel { height: auto; min-height: 5; border: round #263449; padding: 1; margin-bottom: 1; }
    #memory-details { height: auto; min-height: 3; }
    #subagent-preview { height: 7; overflow-y: auto; }
    #settings-panel { padding-bottom: 1; }
    #token-status, #usage-status, #subagent-token { height: auto; }
    Footer { background: #111827; }
    """

    def __init__(self, url: str, workspace: Path) -> None:
        super().__init__()
        self.url = url
        self.workspace = workspace.expanduser().resolve()
        self.client: InferenceClient | None = None
        self.main_agent: Any = None
        self.model_name = "connecting"
        self._requested_context_limit = _env_context_limit()
        self.context_capacity = self._requested_context_limit
        self._context_input_dirty = False
        self._programmatic_context_value: str | None = None
        self._updating_context_inputs = False
        self._runtime_status: dict[str, Any] = {}
        self._runtime_memory: dict[str, Any] = {}
        self._busy = False
        self._current_output = ""
        self._approval_counter = 0
        self._usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._last_context: dict[str, Any] = {}
        self._active_subagent = False
        self._current_settings: dict[str, Any] = {}
        self._current_tool_agent = "MAIN"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="main-pane"):
                yield Static("WORKSPACE / CONVERSATION", classes="section-title")
                yield RichLog(
                    id="conversation", wrap=True, markup=False, highlight=False
                )
                yield Static("Waiting for a prompt.", id="live-output", markup=False)
                with Vertical(id="composer"):
                    yield TextArea(
                        "",
                        id="prompt",
                        show_line_numbers=False,
                        soft_wrap=True,
                    )
                    with Horizontal(id="buttons"):
                        yield Button(
                            "Send  F2 / Ctrl+Enter", id="send", variant="primary"
                        )
                        yield Button("Clear history", id="clear")
                        yield Button("Refresh runtime  F5", id="refresh")
                        yield Button("Quit", id="quit", variant="error")
            with Vertical(id="side-pane"):
                yield Static("RUNTIME", classes="section-title")
                with Vertical(id="runtime-panel"):
                    yield Static(
                        "Connecting to inference runtime…",
                        id="runtime-details",
                        markup=False,
                    )
                yield Static("MEMORY / RESOURCE USAGE", classes="section-title")
                with Vertical(id="memory-panel"):
                    yield Static(
                        "Waiting for runtime memory metrics…",
                        id="memory-details",
                        markup=False,
                    )
                yield Static("MAIN AGENT", classes="section-title")
                with Vertical(id="agent-panel"):
                    yield Static("State: INITIALIZING", id="main-state")
                    yield Static("History: 0 messages", id="history-state")
                yield Static("TOKEN / CONTEXT CONTROL", classes="section-title")
                with Vertical(id="token-panel"):
                    yield Static(
                        "No request inspected yet.", id="token-status", markup=False
                    )
                    yield Static(
                        "Usage: waiting for server token metadata",
                        id="usage-status",
                        markup=False,
                    )
                yield Static("TEMPORARY SUBAGENT", classes="section-title")
                with Vertical(id="subagent-panel"):
                    yield Static("State: IDLE", id="subagent-state")
                    yield Static(
                        "No delegated task.", id="subagent-preview", markup=False
                    )
                    yield Static("Usage: —", id="subagent-token", markup=False)
                yield Static("GENERATION SETTINGS", classes="section-title")
                with Vertical(id="settings-panel"):
                    yield Static("Max output tokens", classes="setting-label")
                    yield Input(value="1024", id="max-output", classes="setting-input")
                    yield Static("Main context limit (tokens)", classes="setting-label")
                    yield Input(
                        value=str(self._requested_context_limit),
                        id="context-limit",
                        classes="setting-input",
                    )
                    yield Static(
                        "Subagent context limit (tokens)", classes="setting-label"
                    )
                    yield Input(
                        value=str(
                            min(DEFAULT_SUBAGENT_CONTEXT, self._requested_context_limit)
                        ),
                        id="subagent-context-limit",
                        classes="setting-input",
                    )
                    yield Static("Temperature", classes="setting-label")
                    yield Input(value="0.2", id="temperature", classes="setting-input")
                    yield Static(
                        "Context overflow (truncate / reject)", classes="setting-label"
                    )
                    yield Input(
                        value="truncate", id="overflow-policy", classes="setting-input"
                    )
                    with Horizontal(id="reasoning-row"):
                        yield Static("Reasoning effort", classes="muted")
                        yield Switch(value=True, id="reasoning")
                    with Horizontal(id="shell-row"):
                        yield Static("Allow shell commands", classes="muted")
                        yield Switch(value=True, id="allow-shell")
        yield Footer()

    def on_mount(self) -> None:
        if not self.workspace.exists():
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.client = InferenceClient(self.url, timeout=180)
        self.main_agent = self._create_main_agent()
        self._current_settings = self._settings()
        self.query_one("#prompt", TextArea).focus()
        self.refresh_runtime_worker()
        self._write_log(f"Workspace: {self.workspace}", "#94a3b8")
        self._write_log(
            "Main agent initialized. Type a coding task; press F2 or Ctrl+Enter to send.",
            "#86efac",
        )

    def on_unmount(self) -> None:
        if self.client is not None:
            self.client.close()

    def _settings(self) -> dict[str, Any]:
        def integer(widget_id: str, fallback: int, minimum: int, maximum: int) -> int:
            try:
                value = int(self.query_one(f"#{widget_id}", Input).value.strip())
            except (ValueError, AttributeError):
                value = fallback
            return max(minimum, min(maximum, value))

        try:
            temperature = float(self.query_one("#temperature", Input).value.strip())
        except (ValueError, AttributeError):
            temperature = 0.2
        temperature = max(0.0, min(2.0, temperature))

        overflow = self.query_one("#overflow-policy", Input).value.strip().lower()
        if overflow not in {"truncate", "reject"}:
            overflow = "truncate"

        context_limit = integer(
            "context-limit", self.context_capacity, 256, self.context_capacity
        )
        subagent_context_limit = integer(
            "subagent-context-limit",
            min(DEFAULT_SUBAGENT_CONTEXT, context_limit),
            256,
            context_limit,
        )
        # Keep a small operational margin. The server's native token inspection
        # remains authoritative because its chat template adds model-specific tokens.
        output_ceiling = max(64, context_limit - 256)
        max_output_tokens = integer("max-output", 1024, 64, min(8192, output_ceiling))

        self._set_context_input("context-limit", context_limit)
        self._set_context_input("subagent-context-limit", subagent_context_limit)
        self.query_one("#max-output", Input).value = str(max_output_tokens)
        return {
            "max_output_tokens": max_output_tokens,
            "context_limit": context_limit,
            "subagent_context_limit": subagent_context_limit,
            "temperature": temperature,
            "reasoning_effort": "on"
            if self.query_one("#reasoning", Switch).value
            else "none",
            "context_overflow": overflow,
            "allow_shell": self.query_one("#allow-shell", Switch).value,
        }

    def _set_context_input(self, widget_id: str, value: int) -> None:
        """Set a token input without mistaking the resulting event for user input."""
        text_value = str(value)
        if widget_id == "context-limit":
            self._programmatic_context_value = text_value
        self._updating_context_inputs = True
        try:
            self.query_one(f"#{widget_id}", Input).value = text_value
        finally:
            self._updating_context_inputs = False

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "context-limit":
            return
        current = event.input.value
        if self._programmatic_context_value == current:
            self._programmatic_context_value = None
            return
        self._programmatic_context_value = None
        self._context_input_dirty = True

    def _create_main_agent(self):
        assert self.client is not None
        tools = self._workspace_tools() + [self._delegate_tool()]
        return self.client.agent(
            instructions=MAIN_INSTRUCTIONS,
            tools=tools,
            max_iterations=10,
            max_output_tokens=1024,
            context_limit=self._requested_context_limit,
            context_overflow="truncate",
            tool_choice="auto",
            trace_callback=lambda event: self._trace_callback("MAIN", event),
        )

    def _workspace_tools(self) -> list[Any]:
        @tool
        def list_files(path: str = ".", max_items: int = 100) -> dict[str, Any]:
            """List files and directories under the selected workspace. Hidden and dependency directories are skipped."""
            directory = self._resolve_workspace_path(path)
            if not directory.is_dir():
                raise ValueError(f"Not a directory: {path}")
            limit = max(1, min(int(max_items), 300))
            entries: list[dict[str, str]] = []
            for item in sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            ):
                if item.name in IGNORED_DIRS or item.name.startswith("."):
                    continue
                self._resolve_workspace_path(str(item.relative_to(self.workspace)))
                entries.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                    }
                )
                if len(entries) >= limit:
                    break
            return {
                "workspace": str(self.workspace),
                "path": path,
                "entries": entries,
                "truncated": len(entries) >= limit,
            }

        @tool
        def read_file(path: str, max_chars: int = 8000) -> dict[str, Any]:
            """Read a UTF-8 text file inside the selected workspace. Output is size-limited."""
            target = self._resolve_workspace_path(path)
            if not target.is_file():
                raise ValueError(f"Not a file: {path}")
            limit = max(1, min(int(max_chars), MAX_READ_CHARS))
            data = target.read_bytes()
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("File is not UTF-8 text") from exc
            return {
                "path": str(target.relative_to(self.workspace)),
                "content": content[:limit],
                "truncated": len(content) > limit,
                "characters": len(content),
            }

        @tool
        def search_text(
            query: str, path: str = ".", max_matches: int = 25
        ) -> dict[str, Any]:
            """Search UTF-8 text files under a workspace directory, skipping common dependency and VCS folders."""
            if not query:
                raise ValueError("query cannot be empty")
            root = self._resolve_workspace_path(path)
            if not root.is_dir():
                raise ValueError(f"Not a directory: {path}")
            limit = max(1, min(int(max_matches), 50))
            matches: list[dict[str, Any]] = []
            for current, directories, filenames in os.walk(root):
                directories[:] = [
                    name
                    for name in directories
                    if name not in IGNORED_DIRS and not name.startswith(".")
                ]
                for filename in filenames:
                    candidate = Path(current) / filename
                    try:
                        self._resolve_workspace_path(
                            str(candidate.relative_to(self.workspace))
                        )
                        if candidate.stat().st_size > 500_000:
                            continue
                        content = candidate.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError, ValueError):
                        continue
                    for line_number, line in enumerate(content.splitlines(), start=1):
                        if query.casefold() in line.casefold():
                            matches.append(
                                {
                                    "path": str(candidate.relative_to(self.workspace)),
                                    "line": line_number,
                                    "text": line[:300],
                                }
                            )
                            if len(matches) >= limit:
                                return {"matches": matches, "truncated": True}
            return {"matches": matches, "truncated": False}

        @tool
        def write_file(path: str, content: str) -> dict[str, Any]:
            """Create or overwrite a UTF-8 file inside the workspace. Always requires explicit user approval."""
            target = self._resolve_workspace_path(path)
            if len(content) > 1_000_000:
                raise ValueError("File content exceeds the 1 MB tool limit")
            if target.exists() and target.is_dir():
                raise ValueError("Target path is a directory")
            preview = content[:3000] + (
                "\n… [preview truncated]" if len(content) > 3000 else ""
            )
            approved = self._request_approval(
                "Approve workspace file write",
                f"Path: {target.relative_to(self.workspace)}\nCharacters: {len(content)}\n\n{preview}",
                "This will create or overwrite the specified workspace file.",
            )
            if not approved:
                return {
                    "ok": False,
                    "denied": True,
                    "path": path,
                    "message": "User denied this write. Do not retry.",
                }
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-resolve after directory creation to catch a changed/symlinked parent.
            target = self._resolve_workspace_path(path)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=target.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            return {
                "ok": True,
                "path": str(target.relative_to(self.workspace)),
                "characters_written": len(content),
            }

        @tool
        def run_command(command: str, timeout_seconds: int = 90) -> dict[str, Any]:
            """Run a shell command with the workspace as its working directory. Requires shell commands to be enabled and explicit approval. NOT sandboxed; command has the user's OS permissions."""
            settings = dict(self._current_settings)
            if not settings["allow_shell"]:
                return {
                    "ok": False,
                    "denied": True,
                    "message": "Shell commands are disabled in the UI. Ask the user to enable them; do not retry automatically.",
                }
            if not command.strip() or len(command) > 3000:
                raise ValueError("Command must contain 1–3000 characters")
            timeout = max(1, min(int(timeout_seconds), 180))
            approved = self._request_approval(
                "Approve shell command",
                f"Working directory: {self.workspace}\nTimeout: {timeout}s\n\n{command}",
                "WARNING: This is NOT a sandbox. The command runs with your user permissions and can access files outside the workspace.",
            )
            if not approved:
                return {
                    "ok": False,
                    "denied": True,
                    "message": "User denied this command. Do not retry.",
                }
            try:
                completed = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-lc", command],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = completed.stdout[-MAX_TOOL_OUTPUT:]
                stderr = completed.stderr[-MAX_TOOL_OUTPUT:]
                return {
                    "ok": completed.returncode == 0,
                    "exit_code": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": str(self.workspace),
                }
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return {
                    "ok": False,
                    "timed_out": True,
                    "stdout": str(stdout)[-MAX_TOOL_OUTPUT:],
                    "stderr": str(stderr)[-MAX_TOOL_OUTPUT:],
                    "timeout_seconds": timeout,
                }

        return [list_files, read_file, search_text, write_file, run_command]

    def _delegate_tool(self):
        @tool
        def delegate_to_subagent(
            task: str, focus: str = "implementation"
        ) -> dict[str, Any]:
            """Delegate one bounded coding, code-review, debugging or test-analysis task to a temporary subagent. The subagent returns a report; its local history is discarded afterwards."""
            return self._run_subagent(task, focus, parent_agent="MAIN")

        return delegate_to_subagent

    def _run_subagent(
        self, task: str, focus: str, *, parent_agent: str = "MAIN"
    ) -> dict[str, Any]:
        if self._active_subagent:
            return {
                "ok": False,
                "error": "A subagent is already active. Wait for it to finish.",
            }
        if not task.strip():
            return {"ok": False, "error": "Subagent task cannot be empty."}
        self._active_subagent = True
        subagent_id = f"sub-{uuid.uuid4().hex[:7]}"
        settings = dict(self._current_settings)
        subagent_context_limit = min(
            settings.get("subagent_context_limit", DEFAULT_SUBAGENT_CONTEXT),
            self.context_capacity,
            settings["context_limit"],
        )
        settings["context_limit"] = max(256, subagent_context_limit)
        settings["max_output_tokens"] = min(
            settings["max_output_tokens"], 768, max(64, settings["context_limit"] - 256)
        )
        output: list[str] = []
        subagent = None
        succeeded = False
        started = time.perf_counter()
        if parent_agent == "MAIN":
            self.call_from_thread(
                self._set_agent_state, "MAIN", "PAUSED · SUBAGENT RUNNING"
            )
        else:
            self.call_from_thread(self._set_agent_state, "MAIN", "IDLE · USER SUBAGENT")
        self._current_tool_agent = "SUBAGENT"
        self.call_from_thread(
            self._set_subagent_state,
            "RUNNING",
            f"{subagent_id}\nFocus: {focus}\nTask: {task}",
        )
        try:
            assert self.client is not None
            subagent = self.client.agent(
                instructions=SUBAGENT_INSTRUCTIONS,
                tools=self._workspace_tools(),
                max_iterations=10,
                max_output_tokens=settings["max_output_tokens"],
                context_limit=settings["context_limit"],
                context_overflow=settings["context_overflow"],
                tool_choice="auto",
                trace_callback=lambda event: self._trace_callback("SUBAGENT", event),
            )
            self._inspect_context("SUBAGENT", subagent, task, settings)
            for event in subagent.stream(
                task,
                trace=True,
                max_output_tokens=settings["max_output_tokens"],
                context_limit=settings["context_limit"],
                context_overflow=settings["context_overflow"],
                temperature=settings["temperature"],
                reasoning_effort=settings["reasoning_effort"],
            ):
                if isinstance(event, StreamEvent):
                    if event.text:
                        output.append(event.text)
                        preview = "".join(output)[-3500:]
                        self.call_from_thread(self._set_subagent_preview, preview)
                    self._capture_usage("SUBAGENT", event)
                elif isinstance(event, AgentEvent):
                    continue
            result = (
                subagent.last_result.text
                if subagent.last_result
                else "".join(output).strip()
            )
            succeeded = bool(result.strip())
            report = {
                "ok": succeeded,
                "subagent_id": subagent_id,
                "focus": focus,
                "result": result[:6000],
                "duration_seconds": round(time.perf_counter() - started, 2),
                "note": "Subagent local history was discarded after returning this result.",
            }
            self.call_from_thread(
                self._set_subagent_state,
                "COMPLETED",
                f"{subagent_id}\nFocus: {focus}\nCompleted in {report['duration_seconds']}s",
            )
            self.call_from_thread(
                self._set_subagent_preview,
                result[:3500] or "Subagent returned no text.",
            )
            return report
        except Exception as exc:
            self.call_from_thread(
                self._set_subagent_state, "FAILED", f"{subagent_id}\n{exc}"
            )
            return {"ok": False, "subagent_id": subagent_id, "error": str(exc)}
        finally:
            if subagent is not None:
                try:
                    subagent.reset()
                except Exception:
                    pass
                del subagent
            self._active_subagent = False
            self._current_tool_agent = "MAIN"
            if parent_agent == "MAIN":
                parent_state = (
                    "RESUMING · SUBAGENT COMPLETED"
                    if succeeded
                    else "RESUMING · SUBAGENT FAILED"
                )
                self.call_from_thread(self._set_agent_state, "MAIN", parent_state)

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        supplied = Path(raw_path)
        # if supplied.is_absolute():
        #    raise ValueError("Use workspace-relative paths, not absolute paths")
        candidate = (self.workspace / supplied).resolve(strict=False)
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("Path escapes the selected workspace") from exc
        return candidate

    def _request_approval(self, title: str, detail: str, warning: str = "") -> bool:
        self._approval_counter += 1
        self.call_from_thread(
            self._set_agent_state, self._current_tool_agent, "WAITING FOR USER APPROVAL"
        )
        result: Future[bool] = Future()
        try:
            self.call_from_thread(self._push_approval, title, detail, warning, result)
            return result.result()
        except Exception:
            return False

    def _push_approval(
        self, title: str, detail: str, warning: str, result: Future[bool]
    ) -> None:
        def completed(approved: bool | None) -> None:
            if not result.done():
                result.set_result(bool(approved))

        self.push_screen(ApprovalScreen(title, detail, warning), callback=completed)

    def _trace_callback(self, agent_name: str, event: AgentEvent) -> None:
        if event.type == "tool_call_started":
            self._current_tool_agent = agent_name
            name = str(event.data.get("name", "tool"))
            arguments = json.dumps(
                event.data.get("arguments", {}), ensure_ascii=False, default=str
            )
            self.call_from_thread(self._log_tool, agent_name, name, arguments)
            self.call_from_thread(self._set_agent_state, agent_name, f"TOOL · {name}")
        elif event.type == "tool_call_completed":
            name = str(event.data.get("name", "tool"))
            success = bool(event.data.get("success", False))
            duration = float(event.data.get("duration_ms", 0.0))
            self.call_from_thread(
                self._log_tool_result,
                agent_name,
                name,
                success,
                duration,
                event.data.get("error"),
            )
            self.call_from_thread(self._set_agent_state, agent_name, "RUNNING")

    def _inspect_context(
        self, agent_name: str, agent: Any, prompt: str, settings: dict[str, Any]
    ) -> None:
        assert self.client is not None
        messages = agent.messages + [{"role": "user", "content": prompt}]
        definitions = [spec.definition() for spec in agent.tools.values()]
        context = self.client.chat.context(
            messages,
            tools=definitions,
            tool_choice="auto" if definitions else None,
            max_output_tokens=settings["max_output_tokens"],
            context_limit=settings["context_limit"],
            context_overflow=settings["context_overflow"],
            temperature=settings["temperature"],
            reasoning_effort=settings["reasoning_effort"],
        )
        window = context.context_window or settings["context_limit"]
        prompt_tokens = context.prompt_tokens
        safety_margin = int(getattr(context, "safety_margin_tokens", 0) or 16)
        remaining = context.remaining_tokens
        if remaining is None and prompt_tokens is not None:
            remaining = max(
                0,
                window - prompt_tokens - settings["max_output_tokens"] - safety_margin,
            )
        data = {
            "prompt_tokens": prompt_tokens,
            "context_window": window,
            "remaining_tokens": remaining,
            "max_output_tokens": settings["max_output_tokens"],
            "safety_margin_tokens": safety_margin,
            "configured_context_limit": settings["context_limit"],
            "fits": context.fits,
            "context_overflow": settings["context_overflow"],
        }
        self.call_from_thread(self._update_context, agent_name, data)
        if context.fits is False and settings["context_overflow"] == "reject":
            raise ValueError(
                f"Prompt requires {prompt_tokens} prompt tokens plus "
                f"{settings['max_output_tokens']} output tokens, exceeding the "
                f"{window}-token context limit. Increase the context limit/output "
                "budget or select context overflow='truncate'."
            )

    def _capture_usage(self, agent_name: str, event: StreamEvent) -> None:
        usage = event.data.get("usage") if isinstance(event.data, dict) else None
        if not isinstance(usage, dict) or not usage:
            return
        # Some OpenAI-compatible SSE implementations put usage on the final
        # chunk with choices=[], so do not require finish_reason here.
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        normalized = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        }
        self._usage_totals["prompt_tokens"] += normalized["prompt_tokens"]
        self._usage_totals["completion_tokens"] += normalized["completion_tokens"]
        self._usage_totals["total_tokens"] += normalized["total_tokens"]
        self.call_from_thread(
            self._update_usage, agent_name, normalized, dict(self._usage_totals)
        )

    def _write_log(self, text: str, color: str = "#cbd5e1") -> None:
        log = self.query_one("#conversation", RichLog)
        from rich.text import Text

        log.write(Text(text, style=color))

    def _begin_turn(self, prompt: str) -> None:
        self._current_output = ""
        log = self.query_one("#conversation", RichLog)
        from rich.text import Text

        log.write(Text("YOU  › " + prompt, style="bold #93c5fd"))
        self.query_one("#live-output", Static).update("ARC is working…")
        self._set_agent_state("MAIN", "INSPECTING CONTEXT")
        self.query_one("#send", Button).disabled = True
        self.query_one("#prompt", TextArea).disabled = True
        for selector in (
            "#max-output",
            "#context-limit",
            "#subagent-context-limit",
            "#temperature",
            "#overflow-policy",
            "#reasoning",
            "#allow-shell",
        ):
            self.query_one(selector).disabled = True

    def _append_live(self, accumulated: str) -> None:
        self._current_output = accumulated
        self.query_one("#live-output", Static).update(accumulated or "ARC is working…")

    def _end_turn(self, result: str, error: str | None = None) -> None:
        from rich.text import Text

        log = self.query_one("#conversation", RichLog)
        if result:
            log.write(Text("ARC", style="bold #86efac"))
            log.write(Text(result, style="#d1fae5"))
        elif error:
            log.write(Text(f"REQUEST FAILED · {error}", style="bold #fca5a5"))
        self.query_one("#live-output", Static).update("Ready for the next task.")
        self.query_one("#send", Button).disabled = False
        self.query_one("#prompt", TextArea).disabled = False
        for selector in (
            "#max-output",
            "#context-limit",
            "#subagent-context-limit",
            "#temperature",
            "#overflow-policy",
            "#reasoning",
            "#allow-shell",
        ):
            self.query_one(selector).disabled = False
        self.query_one("#prompt", TextArea).focus()
        self._set_agent_state("MAIN", "IDLE")
        self._update_history_count()

    def _set_agent_state(self, agent_name: str, state: str) -> None:
        widget_id = "#main-state" if agent_name == "MAIN" else "#subagent-state"
        label = "Main" if agent_name == "MAIN" else "Subagent"
        self.query_one(widget_id, Static).update(f"{label}: {state}")

    def _set_subagent_state(self, state: str, details: str) -> None:
        self.query_one("#subagent-state", Static).update(f"Subagent: {state}")
        self.query_one("#subagent-preview", Static).update(details)

    def _set_subagent_preview(self, text: str) -> None:
        self.query_one("#subagent-preview", Static).update(
            text or "Subagent is working…"
        )

    def _log_tool(self, agent_name: str, name: str, arguments: str) -> None:
        self._write_log(f"{agent_name} TOOL → {name} {arguments[:1500]}", "#fcd34d")

    def _log_tool_result(
        self, agent_name: str, name: str, success: bool, duration_ms: float, error: Any
    ) -> None:
        status = "OK" if success else f"FAILED: {error}"
        self._write_log(
            f"{agent_name} TOOL ← {name} · {status} · {duration_ms:.0f} ms",
            "#86efac" if success else "#fca5a5",
        )

    def _update_context(self, agent_name: str, data: dict[str, Any]) -> None:
        self._last_context[agent_name] = data
        prompt = data.get("prompt_tokens")
        window = data.get("context_window") or data.get("configured_context_limit")
        remaining = data.get("remaining_tokens")
        output = data.get("max_output_tokens", 0)
        margin = data.get("safety_margin_tokens", 0)
        fits = data.get("fits")
        if agent_name == "MAIN":
            self.query_one("#token-status", Static).update(
                f"Context: {self._fmt_int(window)} tokens  |  prompt: {self._fmt_int(prompt)}\n"
                f"Output reserved: {self._fmt_int(output)}  |  safety margin: {self._fmt_int(margin)}\n"
                f"Available after reservations: {self._fmt_int(remaining)} tokens\n"
                f"Prompt fit: {fits if fits is not None else 'not reported'}  |  policy: {data.get('context_overflow', '?')}\n"
                f"{self._bar(prompt, window)}"
            )
        else:
            self.query_one("#subagent-token", Static).update(
                f"Context: {self._fmt_int(window)} tokens\n"
                f"Prompt: {self._fmt_int(prompt)}  |  available: {self._fmt_int(remaining)}\n"
                f"Output reserved: {self._fmt_int(output)}  |  fit: {fits if fits is not None else 'not reported'}"
            )

    @staticmethod
    def _fmt_int(value: Any) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _bar(value: Any, total: Any, width: int = 24) -> str:
        try:
            fraction = max(0.0, min(1.0, float(value) / float(total)))
        except (TypeError, ValueError, ZeroDivisionError):
            return "[context usage unavailable]"
        filled = round(fraction * width)
        return "[" + "█" * filled + "░" * (width - filled) + f"] {fraction:.0%}"

    def _update_usage(
        self, agent_name: str, usage: dict[str, int], totals: dict[str, int]
    ) -> None:
        self.query_one("#usage-status", Static).update(
            f"Latest {agent_name}: {self._fmt_int(usage['prompt_tokens'])} prompt / "
            f"{self._fmt_int(usage['completion_tokens'])} generated\n"
            f"Session total: {self._fmt_int(totals['prompt_tokens'])} prompt / "
            f"{self._fmt_int(totals['completion_tokens'])} generated / "
            f"{self._fmt_int(totals['total_tokens'])} total"
        )
        context = self._last_context.get(agent_name, {})
        window = context.get("context_window") or context.get(
            "configured_context_limit"
        )
        output_reserve = int(context.get("max_output_tokens", 0) or 0)
        safety_margin = int(context.get("safety_margin_tokens", 16) or 16)
        remaining = None
        if window is not None:
            remaining = max(
                0, int(window) - usage["prompt_tokens"] - output_reserve - safety_margin
            )
        if agent_name == "MAIN":
            self.query_one("#token-status", Static).update(
                f"Context: {self._fmt_int(window)} tokens  |  latest prompt: "
                f"{self._fmt_int(usage['prompt_tokens'])}\n"
                f"Output reserved: {self._fmt_int(output_reserve)}  |  margin: {self._fmt_int(safety_margin)}\n"
                f"Available after reservations: {self._fmt_int(remaining)} tokens\n"
                f"{self._bar(usage['prompt_tokens'], window)}"
            )
        else:
            self.query_one("#subagent-token", Static).update(
                f"Latest prompt: {self._fmt_int(usage['prompt_tokens'])} / {self._fmt_int(window)}\n"
                f"Available after reservations: {self._fmt_int(remaining)}\n"
                f"Generated: {self._fmt_int(usage['completion_tokens'])} tokens"
            )

    def _update_history_count(self) -> None:
        if self.main_agent is not None:
            self.query_one("#history-state", Static).update(
                f"History: {len(self.main_agent.messages)} local messages"
            )

    @work(thread=True, exclusive=True, group="agent")
    def _run_prompt_worker(self, prompt: str, settings: dict[str, Any]) -> None:
        try:
            self.call_from_thread(self._begin_turn, prompt)
            self._inspect_context("MAIN", self.main_agent, prompt, settings)
            output: list[str] = []
            self.call_from_thread(self._set_agent_state, "MAIN", "GENERATING")
            for event in self.main_agent.stream(
                prompt,
                trace=True,
                max_output_tokens=settings["max_output_tokens"],
                context_limit=settings["context_limit"],
                context_overflow=settings["context_overflow"],
                temperature=settings["temperature"],
                reasoning_effort=settings["reasoning_effort"],
            ):
                if isinstance(event, StreamEvent):
                    if event.text:
                        output.append(event.text)
                        self.call_from_thread(
                            self._append_live, "".join(output)[-20_000:]
                        )
                    if event.reasoning:
                        self.call_from_thread(
                            self._set_agent_state, "MAIN", "REASONING"
                        )
                    if event.tool_calls:
                        names = ", ".join(
                            call.name or "tool" for call in event.tool_calls
                        )
                        self.call_from_thread(
                            self._set_agent_state, "MAIN", f"REQUESTING TOOL · {names}"
                        )
                    self._capture_usage("MAIN", event)
            result = (
                self.main_agent.last_result.text
                if self.main_agent.last_result
                else "".join(output).strip()
            )
            self.call_from_thread(self._end_turn, result)
        except Exception as exc:
            self.call_from_thread(self._end_turn, "", str(exc))
        finally:
            self._busy = False

    @work(thread=True, exclusive=True, group="agent")
    def _run_direct_subagent_worker(self, task: str) -> None:
        try:
            self.call_from_thread(self._begin_turn, f"/delegate {task}")
            result = self._run_subagent(task, "user-requested", parent_agent="USER")
            rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            self.call_from_thread(self._end_turn, rendered)
        except Exception as exc:
            self.call_from_thread(self._end_turn, "", str(exc))
        finally:
            self._busy = False

    @work(thread=True, exclusive=True, group="runtime")
    def refresh_runtime_worker(self) -> None:
        if self.client is None:
            return
        try:
            model = self.client.model.get()
            status = self.client.runtime.status()
            try:
                memory = self.client.runtime.memory()
            except Exception as exc:
                memory = {"error": str(exc)}
            self.call_from_thread(
                self._update_runtime,
                model.id or "unknown",
                model.supports_tools,
                model.tool_protocol,
                status,
                memory,
            )
        except Exception as exc:
            self.call_from_thread(self._update_runtime_error, str(exc))

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass
        return {"value": value}

    @staticmethod
    def _find_nested(data: Any, names: tuple[str, ...]) -> Any:
        wanted = {name.casefold() for name in names}
        if isinstance(data, dict):
            for key, value in data.items():
                if str(key).casefold() in wanted and value is not None:
                    return value
            for value in data.values():
                found = CodingAgentApp._find_nested(value, names)
                if found is not None:
                    return found
        elif isinstance(data, (list, tuple)):
            for value in data:
                found = CodingAgentApp._find_nested(value, names)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _flatten_metrics(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    rows.extend(CodingAgentApp._flatten_metrics(value, label))
                elif isinstance(value, (list, tuple)):
                    rendered = ", ".join(str(item) for item in value[:6])
                    if len(value) > 6:
                        rendered += ", …"
                    rows.append((label, rendered))
                else:
                    rows.append((label, value))
        elif data is not None:
            rows.append((prefix or "value", data))
        return rows

    @staticmethod
    def _format_metric(key: str, value: Any) -> str:
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif value is None:
            rendered = "—"
        elif isinstance(value, float):
            rendered = f"{value:,.2f}"
        elif isinstance(value, int):
            if "byte" in key.casefold():
                gib = value / (1024**3)
                rendered = f"{value:,} B ({gib:.2f} GiB)"
            else:
                rendered = f"{value:,}"
        else:
            rendered = str(value).replace("\n", " ")
        return rendered

    def _update_runtime(
        self,
        model: str,
        tools: bool,
        protocol: Any,
        status: dict[str, Any],
        memory: dict[str, Any],
    ) -> None:
        self.model_name = model
        status_map = self._mapping(status)
        memory_map = self._mapping(memory)
        self._runtime_status = status_map
        self._runtime_memory = memory_map

        reported_context = self._find_nested(
            status_map,
            (
                "context_window",
                "n_ctx",
                "context_length",
                "context_limit",
                "configured_context",
            ),
        )
        try:
            if reported_context is not None:
                self.context_capacity = max(256, int(reported_context))
        except (TypeError, ValueError):
            pass

        # Adopt the server's actual context capacity on first connection. After
        # the user edits the field, preserve their value while clamping to capacity.
        context_input = self.query_one("#context-limit", Input)
        try:
            current_context = int(context_input.value.strip())
        except (ValueError, AttributeError):
            current_context = self._requested_context_limit
        if not self._context_input_dirty:
            selected_context = self.context_capacity
        else:
            selected_context = min(current_context, self.context_capacity)
        selected_context = max(256, selected_context)
        self._set_context_input("context-limit", selected_context)

        subagent_input = self.query_one("#subagent-context-limit", Input)
        try:
            current_subagent = int(subagent_input.value.strip())
        except (ValueError, AttributeError):
            current_subagent = DEFAULT_SUBAGENT_CONTEXT
        selected_subagent = min(
            current_subagent, selected_context, self.context_capacity
        )
        selected_subagent = max(256, selected_subagent)
        self._set_context_input("subagent-context-limit", selected_subagent)

        state = self._find_nested(
            status_map, ("state", "runtime_state", "service_state", "health", "phase")
        )
        if isinstance(state, (dict, list, tuple)):
            state = None
        if state is None:
            candidate_status = status_map.get("status")
            if isinstance(candidate_status, (str, bool)):
                state = candidate_status
        ready = self._find_nested(status_map, ("ready", "is_ready"))
        if state is None:
            state_text = (
                "READY"
                if ready is True
                else ("NOT READY" if ready is False else "reported")
            )
        else:
            state_text = str(state).upper()
        runtime_lines = [
            f"Service: {state_text}",
            f"Model: {model}",
            f"Context capacity: {self._fmt_int(self.context_capacity)} tokens",
            f"Tool support: {'yes' if tools else 'no'}",
            f"Tool protocol: {protocol if protocol is not None else '—'}",
            f"Endpoint: {self.url}",
        ]
        status_rows = self._flatten_metrics(status_map)
        if status_rows:
            runtime_lines.append("Status metrics:")
            for key, value in status_rows[:8]:
                runtime_lines.append(f"  {key}: {self._format_metric(key, value)}")
            if len(status_rows) > 8:
                runtime_lines.append(f"  … {len(status_rows) - 8} more fields")
        self.query_one("#runtime-details", Static).update("\n".join(runtime_lines))

        memory_rows = self._flatten_metrics(memory_map)
        if memory_rows:
            memory_lines = [
                f"{key}: {self._format_metric(key, value)}"
                for key, value in memory_rows[:14]
            ]
            if len(memory_rows) > 14:
                memory_lines.append(f"… {len(memory_rows) - 14} more metrics")
        else:
            memory_lines = ["No memory metrics returned by the runtime."]
        self.query_one("#memory-details", Static).update("\n".join(memory_lines))

        self.sub_title = (
            f"{model} · n_ctx {self._fmt_int(self.context_capacity)} · "
            f"{self.url} · {self.workspace.name}"
        )
        self._write_log(
            f"Runtime connected: {model}, context {self._fmt_int(self.context_capacity)} tokens.",
            "#86efac",
        )

    def _update_runtime_error(self, error: str) -> None:
        self.query_one("#runtime-details", Static).update(
            f"Runtime unavailable\n{error}"
        )
        self.query_one("#memory-details", Static).update(
            "Memory metrics unavailable until runtime connects."
        )

    def _handle_command(self, prompt: str) -> bool:
        command, _, args = prompt.strip().partition(" ")
        command = command.lower()
        if command == "/clear":
            if self.main_agent is not None:
                self.main_agent.reset()
            self.query_one("#conversation", RichLog).clear()
            self.query_one("#live-output", Static).update(
                "Local main-agent history cleared."
            )
            self._update_history_count()
            self._write_log(
                "Main-agent transcript cleared. No server conversation existed.",
                "#94a3b8",
            )
            return True
        if command == "/status":
            self.refresh_runtime_worker()
            self._write_log("Refreshing runtime status and memory…", "#94a3b8")
            return True
        if command == "/help":
            self._write_log(
                "Commands: /clear · /status · /reasoning on|off · /delegate <task> · /help\nKeys: F2 send · F6 clear · F5 refresh · Ctrl+Q quit (Ctrl+Enter / Ctrl+L also available)",
                "#94a3b8",
            )
            return True
        if command == "/delegate":
            if not args.strip():
                self._write_log("Usage: /delegate <self-contained task>", "#fca5a5")
                return True
            if self._busy:
                self._write_log("An agent task is already running.", "#fca5a5")
                return True
            self._current_settings = self._settings()
            self._busy = True
            self._run_direct_subagent_worker(args.strip())
            return True
        if command == "/reasoning":
            value = args.strip().lower()
            switch = self.query_one("#reasoning", Switch)
            if value in {"on", "true", "1"}:
                switch.value = True
            elif value in {"off", "false", "0"}:
                switch.value = False
            else:
                switch.value = not switch.value
            self._write_log(
                f"Reasoning effort: {'on' if switch.value else 'none'}", "#c4b5fd"
            )
            return True
        return False

    def action_send_prompt(self) -> None:
        if self._busy:
            return
        prompt_widget = self.query_one("#prompt", TextArea)
        prompt = prompt_widget.text.strip()
        if not prompt:
            return
        prompt_widget.clear()
        if prompt.startswith("/") and self._handle_command(prompt):
            return
        self._current_settings = self._settings()
        self._busy = True
        self._run_prompt_worker(prompt, dict(self._current_settings))

    def action_clear_history(self) -> None:
        if self._busy:
            self._write_log(
                "Cannot clear history while an agent turn is running.", "#fca5a5"
            )
            return
        self._handle_command("/clear")

    def action_refresh_runtime(self) -> None:
        self.refresh_runtime_worker()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self.action_send_prompt()
        elif event.button.id == "clear":
            self.action_clear_history()
        elif event.button.id == "refresh":
            self.action_refresh_runtime()
        elif event.button.id == "quit":
            self.exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC V2 Textual coding-agent UI")
    parser.add_argument("--url", default=DEFAULT_URL, help="Inference runtime URL")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project directory accessible to coding tools",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CodingAgentApp(args.url, args.workspace).run()


if __name__ == "__main__":
    main()
