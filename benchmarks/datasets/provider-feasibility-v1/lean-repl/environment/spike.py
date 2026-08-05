"""Measure the pinned Lean REPL as an exploratory goal-state transport."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    ToolInteractiveCommand,
    ToolInteractiveRequest,
    run_tool_command,
)

TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "CONJUNCTION-DECOMPOSITION",
        "command": "example (P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q := by sorry",
        "tactics": ("constructor", "exact hP", "exact hQ"),
        "expected_first_goal_count": 2,
    },
    {
        "task_id": "LOCAL-PREMISE-APPLICATION",
        "command": "example (P Q : Prop) (hP : P) (h : P → Q) : Q := by sorry",
        "tactics": ("exact h hP",),
        "expected_first_goal_count": 0,
    },
)
_LEAN_BIN = "/opt/provider/lean-4.31.0-linux/bin"
_GIT_EXECUTABLE = "/usr/bin/git"
_REPL_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "PATH": _LEAN_BIN,
}
_GIT_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "PATH": "/usr/bin:/bin",
}


class ReplSpikeError(RuntimeError):
    """The pinned checkout or REPL protocol did not match the spike contract."""


def _exchange(
    process: ToolInteractiveCommand,
    request: Mapping[str, object],
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    process.send(json.dumps(request, sort_keys=True))
    response_text = process.read_response()
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ReplSpikeError("Lean REPL response is not valid JSON") from exc
    if not isinstance(response, dict):
        raise ReplSpikeError("Lean REPL response must be a JSON object")
    return response, time.monotonic() - started


def _response_errors(response: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    message = response.get("message")
    if isinstance(message, str):
        errors.append(message)
    messages = response.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, Mapping) or item.get("severity") != "error":
                continue
            data = item.get("data")
            errors.append(data if isinstance(data, str) else repr(item))
    return errors


def run_tasks(repl: Path) -> dict[str, Any]:
    started = time.monotonic()
    request = ToolInteractiveRequest(
        executable=str(repl),
        environment=_REPL_ENVIRONMENT,
        cwd=str(Path.cwd()),
        startup_timeout_seconds=30.0,
        read_timeout_seconds=30.0,
        shutdown_timeout_seconds=5.0,
    )
    command = ToolInteractiveCommand(request)
    try:
        command.start()
    except OSError as exc:
        raise ReplSpikeError("The Lean REPL could not be launched") from exc
    task_results: list[dict[str, Any]] = []
    try:
        for task in TASKS:
            command_response, command_seconds = _exchange(
                command,
                {"cmd": task["command"]},
            )
            command_errors = _response_errors(command_response)
            if command_errors:
                raise ReplSpikeError(
                    f"{task['task_id']} command failed: {'; '.join(command_errors)}"
                )
            sorries = command_response.get("sorries")
            if not isinstance(sorries, list) or len(sorries) != 1:
                raise ReplSpikeError(
                    f"{task['task_id']} did not expose one proof state"
                )
            proof_state = sorries[0].get("proofState")
            if not isinstance(proof_state, int):
                raise ReplSpikeError(
                    f"{task['task_id']} returned an invalid proof state"
                )
            traces: list[dict[str, Any]] = []
            for tactic in task["tactics"]:
                response, elapsed = _exchange(
                    command,
                    {"tactic": tactic, "proofState": proof_state},
                )
                response_errors = _response_errors(response)
                if response_errors:
                    raise ReplSpikeError(
                        f"{task['task_id']} tactic failed: "
                        + "; ".join(response_errors)
                    )
                next_state = response.get("proofState")
                goals = response.get("goals")
                if not isinstance(next_state, int) or not isinstance(goals, list):
                    raise ReplSpikeError(
                        f"{task['task_id']} tactic response is malformed"
                    )
                traces.append(
                    {
                        "tactic": tactic,
                        "elapsed_seconds": round(elapsed, 6),
                        "goal_count": len(goals),
                        "error_count": len(response_errors),
                    }
                )
                proof_state = next_state
            task_results.append(
                {
                    "task_id": task["task_id"],
                    "command_seconds": round(command_seconds, 6),
                    "tactics": traces,
                    "completed": traces[-1]["goal_count"] == 0,
                    "decomposition_observed": (
                        traces[0]["goal_count"] == task["expected_first_goal_count"]
                    ),
                }
            )
    finally:
        return_code = command.close()
    stderr = command.stderr.decode("utf-8", "replace")
    return {
        "protocol": "leanprover-community/repl",
        "task_count": len(task_results),
        "completed_count": sum(result["completed"] for result in task_results),
        "parameter_error_count": sum(
            trace["error_count"]
            for result in task_results
            for trace in result["tactics"]
        ),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "return_code": return_code,
        "stderr": stderr,
        "tasks": task_results,
        "limitations": [
            "completed tactic states cannot be replayed into the originating command",
            "the spike measures protocol viability, not agent outcome improvement",
            "final trust still requires lean.check over explicit source",
        ],
    }


def _verify_pin(checkout: Path, pin: Mapping[str, object]) -> None:
    request = ToolCommandRequest(
        executable=_GIT_EXECUTABLE,
        arguments=("rev-parse", "HEAD"),
        environment=_GIT_ENVIRONMENT,
        cwd=str(checkout.resolve(strict=True)),
        timeout_seconds=10.0,
        stdin_bytes=b"",
        stdout_limit_bytes=256,
        stderr_limit_bytes=256,
    )
    result = run_tool_command(request)
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        raise ReplSpikeError("git rev-parse failed inside the pinned checkout")
    commit = result.stdout.decode("utf-8", "replace").strip()
    if commit != pin.get("commit"):
        raise ReplSpikeError(f"checkout commit {commit} differs from frozen pin")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--repl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    pin_path = Path(__file__).with_name("lean_repl_pin.json")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    _verify_pin(args.checkout, pin)
    repl = args.repl or args.checkout / ".lake" / "build" / "bin" / "repl"
    result = {**pin, **run_tasks(repl)}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["completed_count"] == result["task_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
