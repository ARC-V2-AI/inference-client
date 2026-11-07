"""A Downloads-confined bash tool that needs the user's approval for every command.

Design (defence in depth, all of it best-effort -- see the note at the bottom):

1. Static validation of the command string BEFORE the user is asked anything:
   - allowlist of commands (no shells, interpreters, network tools, sudo, ...)
   - only these operators: ;  &&  ||  |  >  >>
   - no $, backticks, braces, parentheses, newlines (no expansion / substitution)
   - every path-like argument (and every glob match) is resolved with realpath
     and must lie inside the root folder, so ``..``, absolute paths, ``~user``
     and symlinks that point outside are all rejected
   - dangerous flags blocked (find -exec/-delete/-L, grep -R, rm --no-preserve-root, ...)
   - rm / rmdir / mv can never target the root folder itself
2. Explicit y/N confirmation shown in the terminal UI.
3. Execution with cwd=root, HOME=root, a minimal environment, no stdin,
   a wall-clock timeout (process group is killed) and a CPU limit.
4. Output is truncated before it goes back to the model.

Configuration (environment variables):
    ARC_CLIENT_BASH_ROOT     folder the tool may touch      (default ~/Downloads)
    ARC_CLIENT_BASH_TIMEOUT  seconds before the command is killed (default 20)
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

ROOT = Path(os.getenv("ARC_CLIENT_BASH_ROOT", "~/Downloads")).expanduser().resolve()
TIMEOUT_S = float(os.getenv("ARC_CLIENT_BASH_TIMEOUT", "20"))
MAX_OUTPUT_CHARS = 8000
MAX_COMMAND_CHARS = 1000

READ_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "file", "stat", "du",
    "pwd", "echo", "basename", "dirname",
}
WRITE_COMMANDS = {"mkdir", "touch", "cp", "mv", "rm", "rmdir"}
ALLOWED_COMMANDS = READ_COMMANDS | WRITE_COMMANDS

# Commands that must never be pointed at the root folder itself.
_NEVER_ROOT = {"rm", "rmdir", "mv"}

_ALLOWED_OPS = {";", "&&", "||", "|", ">", ">>"}
_PUNCT = set("();<>|&")
_FORBIDDEN_CHARS = set("$`{}\n\r\x00")
_GLOB_CHARS = set("*?[")

_BLOCKED_FLAGS: dict[str, set[str]] = {
    "find": {
        "-exec", "-execdir", "-ok", "-okdir", "-delete",
        "-fprint", "-fprint0", "-fprintf", "-fls", "-L", "-H", "-follow",
    },
    "du": {"-L", "--dereference"},
    "cp": {"-L", "--dereference"},
    "rm": {"--no-preserve-root"},
    "grep": {"--dereference-recursive"},
}
_GREP_FOLLOW_SYMLINKS = re.compile(r"-[A-Za-z]*R[A-Za-z]*")

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command inside the user's Downloads folder (that is the "
            "working directory; nothing outside it is reachable). The user must "
            "approve every command before it runs. Allowed commands: "
            + ", ".join(sorted(ALLOWED_COMMANDS))
            + ". Pipes (|), &&, ||, ; and > / >> redirects are allowed. Not allowed: "
            "variables ($), backticks, braces, parentheses, absolute paths or '..' "
            "leaving the folder, sudo, network access, scripts or interpreters. "
            "If a command is rejected or denied, do not retry the same command."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command line to run, e.g. 'ls -la'",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _inside(path: Path, *, allow_root: bool) -> bool:
    return (allow_root and path == ROOT) or ROOT in path.parents


def _check_path_arg(arg: str, *, never_root: bool) -> str | None:
    """Return an error message if `arg` could reach outside ROOT, else None."""
    if arg.startswith("~") and arg != "~" and not arg.startswith("~/"):
        return f"{arg!r}: '~user' expansion is not allowed"

    relative = True
    if arg == "~":
        rest = ""
    elif arg.startswith("~/"):
        rest = arg[2:]  # the shell runs with HOME=ROOT, so ~ means ROOT
    elif os.path.isabs(arg):
        rest, relative = arg, False
    else:
        rest = arg

    root_str = str(ROOT)
    if relative:
        literal = os.path.join(root_str, rest) if rest else root_str
        pattern = os.path.join(glob.escape(root_str), rest) if rest else glob.escape(root_str)
    else:
        literal = pattern = arg

    path_like = "/" in arg or arg.startswith("~") or arg in {".", ".."}
    if any(c in arg for c in _GLOB_CHARS):
        targets = glob.glob(pattern)  # check what the shell would really expand to
    elif path_like or os.path.lexists(literal):
        targets = [literal]
    else:
        return None  # plain new name, will be created inside the cwd

    for target in targets:
        resolved = Path(os.path.realpath(target))
        if str(resolved) == "/dev/null":
            continue
        if not _inside(resolved, allow_root=not never_root):
            return f"{arg!r} resolves outside the allowed folder ({ROOT})"
    return None


def validate(command: str) -> tuple[bool, str]:
    """Static checks. Returns (ok, reason_if_not_ok)."""
    if not command.strip():
        return False, "empty command"
    if len(command) > MAX_COMMAND_CHARS:
        return False, f"command longer than {MAX_COMMAND_CHARS} characters"
    bad = sorted(c for c in _FORBIDDEN_CHARS if c in command)
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        return False, f"forbidden character(s): {shown} (no variables, substitution, braces or newlines)"

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        return False, f"could not parse command: {exc}"

    segments: list[list[str]] = []
    redirect_targets: list[str] = []
    current: list[str] = []
    pending_redirect = False
    last_op: str | None = None

    for tok in tokens:
        if tok and all(c in _PUNCT for c in tok):
            if tok not in _ALLOWED_OPS:
                return False, f"operator {tok!r} is not allowed"
            if tok in {">", ">>"}:
                if not current or pending_redirect:
                    return False, "misplaced redirect"
                pending_redirect = True
            else:
                if not current or pending_redirect:
                    return False, f"nothing to run before {tok!r}"
                segments.append(current)
                current = []
            last_op = tok
            continue
        last_op = None
        if pending_redirect:
            redirect_targets.append(tok)
            pending_redirect = False
        else:
            current.append(tok)

    if pending_redirect:
        return False, "redirect without a target"
    if current:
        segments.append(current)
    elif last_op not in {";", None}:
        return False, f"dangling operator {last_op!r}"
    if not segments:
        return False, "empty command"

    for target in redirect_targets:
        err = _check_path_arg(target, never_root=True)
        if err:
            return False, f"redirect target {err}"

    for seg in segments:
        name, args = seg[0], seg[1:]
        if name not in ALLOWED_COMMANDS:
            return False, (
                f"command {name!r} is not allowed (allowed: {', '.join(sorted(ALLOWED_COMMANDS))})"
            )
        blocked = _BLOCKED_FLAGS.get(name, set())
        for arg in args:
            if arg.startswith("-"):
                if arg in blocked or (name == "grep" and _GREP_FOLLOW_SYMLINKS.fullmatch(arg)):
                    return False, f"{name}: flag {arg} is not allowed"
                if "/" in arg or "~" in arg:
                    return False, f"{arg!r}: pass paths as separate arguments, not inside a flag"
                continue
            err = _check_path_arg(arg, never_root=name in _NEVER_ROOT)
            if err:
                return False, f"{name}: {err}"
    return True, ""


# ---------------------------------------------------------------------------
# UI + execution
# ---------------------------------------------------------------------------


def _bold(text: str) -> str:
    use = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    return f"\033[1;33m{text}\033[0m" if use else text


def confirm(command: str) -> bool:
    print()
    print(_bold("  BASH │ approval needed"))
    print(f"  folder : {ROOT}")
    print(f"  command: {command}")
    try:
        answer = input("  Run this command? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _limits() -> None:  # runs in the child before exec
    if resource is not None:
        cpu = int(TIMEOUT_S) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def _execute(command: str) -> dict[str, Any]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(ROOT),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    timed_out = False
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
            preexec_fn=_limits if resource is not None else None,
        )
        try:
            proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

        def _read(f: Any) -> tuple[str, bool]:
            f.seek(0)
            data = f.read(MAX_OUTPUT_CHARS * 4 + 1)
            text = data.decode("utf-8", errors="replace")
            return text[:MAX_OUTPUT_CHARS], len(text) > MAX_OUTPUT_CHARS

        stdout, out_trunc = _read(out_f)
        stderr, err_trunc = _read(err_f)

    return {
        "ok": proc.returncode == 0 and not timed_out,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": out_trunc or err_trunc,
        "cwd": str(ROOT),
    }


def run_bash(command: str) -> str:
    """Validate, ask the user, run. Always returns a JSON string for the model."""
    if not ROOT.is_dir():
        return json.dumps({"ok": False, "error": f"folder does not exist: {ROOT}"})

    ok, reason = validate(command)
    if not ok:
        print(_bold("  BASH │ rejected: ") + reason)
        return json.dumps({"ok": False, "error": f"command rejected: {reason}"})

    if not confirm(command):
        print(_bold("  BASH │ denied by user"))
        return json.dumps({"ok": False, "error": "The user denied this command."})

    return json.dumps(_execute(command), ensure_ascii=False)


# NOTE: this is a guard rail, not a sandbox. It parses the command text in Python
# and bash parses it again, so a parsing discrepancy is always possible. For real
# isolation, also run the command under bubblewrap/firejail/a container with only
# the Downloads folder mounted writable and no network.
