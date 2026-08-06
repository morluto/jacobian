"""Deliberately simple untrusted process-plugin entrypoints."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def _linux_process_start_time(pid: int) -> str:
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return stat.rsplit(")", 1)[1].split()[19]


def _record_child_identity(child: subprocess.Popen[bytes], pid_marker: str) -> None:
    pathlib.Path(pid_marker).write_text(
        f"{child.pid}:{_linux_process_start_time(child.pid)}",
        encoding="utf-8",
    )


def echo(request: dict[str, Any]) -> dict[str, Any]:
    print("untrusted plugin diagnostic")
    return {"seen": request}


def report_environment(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "secret": os.environ.get("JACOBIAN_TEST_SECRET"),
        "https_proxy": os.environ.get("HTTPS_PROXY"),
    }


def wait_forever(_request: dict[str, Any]) -> dict[str, Any]:
    time.sleep(60)
    return {"unreachable": True}


def exit_without_response(_request: dict[str, Any]) -> dict[str, Any]:
    os._exit(0)


def imitate_source_change(_request: dict[str, Any]) -> dict[str, Any]:
    raise ValueError("plugin source changed during execution")


def emit_large_diagnostic(_request: dict[str, Any]) -> dict[str, Any]:
    print("x" * 4096)
    time.sleep(60)
    return {"status": "otherwise valid"}


def spawn_delayed_child(request: dict[str, Any]) -> dict[str, Any]:
    marker = request["marker"]
    started_marker = request["started_marker"]
    pid_marker = request["pid_marker"]
    ready_marker = f"{pid_marker}.ready"
    delay_seconds = request.get("delay_seconds", 1)
    script = (
        "import pathlib,time;"
        f"pathlib.Path({ready_marker!r}).write_text('ready', encoding='utf-8');"
        f"time.sleep({delay_seconds!r});"
        f"pathlib.Path({marker!r}).write_text('survived', encoding='utf-8')"
    )
    child = subprocess.Popen([sys.executable, "-c", script])
    _record_child_identity(child, pid_marker)
    while not pathlib.Path(ready_marker).exists():
        time.sleep(0.005)
    pathlib.Path(started_marker).write_text("started", encoding="utf-8")
    time.sleep(60)
    return {"unreachable": True}


def spawn_child_then_return(request: dict[str, Any]) -> dict[str, Any]:
    """Exit the worker while a descendant still owns its output pipes."""

    marker = request["marker"]
    pid_marker = request["pid_marker"]
    ready_marker = f"{pid_marker}.ready"
    delay_seconds = request.get("delay_seconds", 1)
    script = (
        "import pathlib,time;"
        f"pathlib.Path({ready_marker!r}).write_text('ready', encoding='utf-8');"
        f"time.sleep({delay_seconds!r});"
        f"pathlib.Path({marker!r}).write_text('survived', encoding='utf-8')"
    )
    child = subprocess.Popen([sys.executable, "-c", script])
    _record_child_identity(child, pid_marker)
    while not pathlib.Path(ready_marker).exists():
        time.sleep(0.005)
    return {"worker": "returned"}


def spawn_detached_child_then_return(request: dict[str, Any]) -> dict[str, Any]:
    marker = request["marker"]
    pid_marker = request["pid_marker"]
    ready_marker = f"{pid_marker}.ready"
    script = (
        "import pathlib,time;"
        f"pathlib.Path({ready_marker!r}).write_text('ready', encoding='utf-8');"
        "time.sleep(1);"
        f"pathlib.Path({marker!r}).write_text('survived', encoding='utf-8')"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _record_child_identity(child, pid_marker)
    while not pathlib.Path(ready_marker).exists():
        time.sleep(0.005)
    return {"worker": "returned"}
