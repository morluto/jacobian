"""Clean-room independent replay of the pinned Lean/REPL proof tasks.

This module is owned by the verifier and lives in the ``tests/`` build context.
It uses only the Python standard library and does not import ``jacobian``,
``benchmarks.*``, or ``environment/spike.py``.  The verifier calls
:func:`run_replay` during verification to independently derive the expected
tactic traces from the pinned Lean 4.31.0 REPL, then compares the result
directly to the agent-submitted report.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any

_REPL = "/opt/provider/repl/.lake/build/bin/repl"
_LEAN_BIN = "/opt/provider/lean-4.31.0-linux/bin"
_ENV = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": _LEAN_BIN}

_TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "CONJUNCTION-DECOMPOSITION",
        "command": "example (P Q : Prop) (hP : P) (hQ : Q) : P \u2227 Q := by sorry",
        "tactics": ("constructor", "exact hP", "exact hQ"),
    },
    {
        "task_id": "LOCAL-PREMISE-APPLICATION",
        "command": "example (P Q : Prop) (hP : P) (h : P \u2192 Q) : Q := by sorry",
        "tactics": ("exact h hP",),
    },
)

_REPLAY_TIMEOUT = 110.0
_SHUTDOWN_TIMEOUT = 5.0
_MAX_RESPONSE_BYTES = 1 << 20  # 1 MiB


def _response_errors(response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    message = response.get("message")
    if isinstance(message, str):
        errors.append(message)
    messages = response.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict) or item.get("severity") != "error":
                continue
            data = item.get("data")
            errors.append(data if isinstance(data, str) else repr(item))
    return errors


class _FrameReader:
    """Bounded buffered reader for blank-line-terminated REPL response frames.

    The Lean REPL protocol delimits response frames with a blank line
    (``\\n\\n``).  This reader retains a ``bytearray`` buffer across
    exchanges so that suffix bytes left after one frame are available for
    the next read.  Each call to :meth:`read_frame` blocks (with a
    monotonic deadline and ``selectors`` readiness checks) until the
    buffer contains ``\\n\\n``, then returns everything before it and
    preserves the remainder.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = bytearray()

    def read_frame(self, timeout: float) -> bytes:
        """Read one ``\\n\\n``-terminated frame, preserving any suffix."""

        delimiter = b"\n\n"
        deadline = time.monotonic() + timeout
        index = self._buffer.find(delimiter)
        if index != -1:
            frame = bytes(self._buffer[:index])
            del self._buffer[: index + len(delimiter)]
            return frame
        sel = selectors.DefaultSelector()
        sel.register(self._fd, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Lean REPL read timed out")
                events = sel.select(timeout=remaining)
                if not events:
                    raise RuntimeError("Lean REPL read timed out")
                chunk = os.read(self._fd, 4096)
                if not chunk:
                    raise RuntimeError("Lean REPL closed stdout before responding")
                self._buffer.extend(chunk)
                if len(self._buffer) > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("Lean REPL response exceeded 1 MiB")
                index = self._buffer.find(delimiter)
                if index != -1:
                    frame = bytes(self._buffer[:index])
                    del self._buffer[: index + len(delimiter)]
                    return frame
        finally:
            sel.close()


def _exchange(
    process: subprocess.Popen[bytes],
    reader: _FrameReader,
    request: dict[str, object],
    deadline: float,
) -> dict[str, Any]:
    # The REPL protocol terminates request frames with a blank line, matching
    # the production spike's ToolInteractiveCommand.send() transport.
    line = (json.dumps(request, sort_keys=True) + "\n\n").encode("utf-8")
    assert process.stdin is not None
    process.stdin.write(line)
    process.stdin.flush()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Lean REPL replay timed out")
    raw = reader.read_frame(remaining)
    response = json.loads(raw.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError("Lean REPL response must be a JSON object")
    return response


def _run_task(
    process: subprocess.Popen[bytes],
    reader: _FrameReader,
    task: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    command_response = _exchange(process, reader, {"cmd": task["command"]}, deadline)
    if _response_errors(command_response):
        raise RuntimeError(f"{task['task_id']} command failed")
    sorries = command_response.get("sorries")
    if not isinstance(sorries, list) or len(sorries) != 1:
        raise RuntimeError(f"{task['task_id']} did not expose one proof state")
    proof_state = sorries[0].get("proofState")
    if not isinstance(proof_state, int):
        raise RuntimeError(f"{task['task_id']} returned an invalid proof state")
    traces: list[dict[str, Any]] = []
    for tactic in task["tactics"]:
        response = _exchange(
            process,
            reader,
            {"tactic": tactic, "proofState": proof_state},
            deadline,
        )
        response_errors = _response_errors(response)
        if response_errors:
            raise RuntimeError(f"{task['task_id']} tactic failed")
        next_state = response.get("proofState")
        goals = response.get("goals")
        if not isinstance(next_state, int) or not isinstance(goals, list):
            raise RuntimeError(f"{task['task_id']} tactic response is malformed")
        traces.append(
            {
                "tactic": tactic,
                "goal_count": len(goals),
                "error_count": len(response_errors),
            }
        )
        proof_state = next_state
    if not traces or traces[-1]["goal_count"] != 0:
        raise RuntimeError(f"{task['task_id']} left goals unfinished")
    return {"task_id": task["task_id"], "tactics": traces}


def run_replay() -> dict[str, Any]:
    """Independently replay the two pinned proof tasks.

    Returns ``{"ok": True, "tasks": [...]}`` on success or
    ``{"ok": False}`` on any failure.
    """

    if not Path(_REPL).is_file() or not os.access(_REPL, os.X_OK):
        return {"ok": False}
    try:
        process = subprocess.Popen(
            [_REPL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_ENV,
        )
    except OSError:
        return {"ok": False}

    assert process.stdout is not None
    reader = _FrameReader(process.stdout.fileno())
    task_results: list[dict[str, Any]] = []
    replay_ok = False
    try:
        deadline = time.monotonic() + _REPLAY_TIMEOUT
        for task in _TASKS:
            task_results.append(_run_task(process, reader, task, deadline))
        replay_ok = True
    except (OSError, ValueError, RuntimeError, KeyError):
        replay_ok = False
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if not replay_ok or process.returncode != 0:
        return {"ok": False}
    return {"ok": True, "tasks": task_results}


def traces_match(submitted: object, replay_result: dict[str, Any]) -> bool:
    """Compare submitted task traces against verifier-owned replay output.

    Returns ``True`` only when the replay succeeded and every submitted
    task and tactic trace matches the replay's independently derived
    ``tactic``, ``goal_count``, and ``error_count``.
    """

    if not isinstance(replay_result, dict) or replay_result.get("ok") is not True:
        return False
    replay_tasks = replay_result.get("tasks")
    if not isinstance(replay_tasks, list) or not isinstance(submitted, list):
        return False
    if len(submitted) != len(replay_tasks):
        return False
    if not all(
        isinstance(item, dict) and type(item.get("task_id")) is str
        for item in submitted
    ):
        return False
    if tuple(item["task_id"] for item in submitted) != tuple(
        item["task_id"] for item in replay_tasks
    ):
        return False
    return all(
        _task_trace_matches(item, replay_task)
        for item, replay_task in zip(submitted, replay_tasks, strict=True)
    )


def _task_trace_matches(task: object, expected_task: dict[str, Any]) -> bool:
    if not isinstance(task, dict) or task.get("task_id") != expected_task["task_id"]:
        return False
    traces = task.get("tactics") if isinstance(task, dict) else None
    expected_traces = expected_task["tactics"]
    if not isinstance(traces, list) or len(traces) != len(expected_traces):
        return False
    return all(
        isinstance(trace, dict)
        and trace.get("tactic") == expected_trace["tactic"]
        and type(trace.get("goal_count")) is int
        and trace.get("goal_count") == expected_trace["goal_count"]
        and type(trace.get("error_count")) is int
        and trace.get("error_count") == expected_trace["error_count"]
        for trace, expected_trace in zip(traces, expected_traces, strict=True)
    )
