from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

import jacobian.process as process_module
from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    bounded_process_cancellation,
    run_bounded_process,
)


def _cancel_after_marker(event: threading.Event, marker: Path) -> threading.Thread:
    def wait_and_cancel() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                event.set()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=wait_and_cancel)
    thread.start()
    return thread


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_pid_exit(pid: int) -> bool:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.01)
    return not _pid_exists(pid)


@pytest.mark.parametrize("timeout_seconds", [math.inf, math.nan])
def test_nonfinite_timeout_is_rejected_before_process_launch(
    timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        run_bounded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            input_bytes=b"",
            timeout_seconds=timeout_seconds,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("prlimit") is None,
    reason="pre-exec resource limits require util-linux prlimit",
)
def test_target_observes_resource_limits_at_startup() -> None:
    prlimit = shutil.which("prlimit")
    assert prlimit is not None
    address_space = 512 * 1024 * 1024
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import json, resource; "
                "print(json.dumps({'cpu': resource.getrlimit(resource.RLIMIT_CPU), "
                "'memory': resource.getrlimit(resource.RLIMIT_AS), "
                "'file': resource.getrlimit(resource.RLIMIT_FSIZE)}))"
            ),
        ],
        input_bytes=b"",
        timeout_seconds=5,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
        resource_limits=ProcessResourceLimits(
            cpu_seconds=2,
            address_space_bytes=address_space,
            file_size_bytes=1024 * 1024,
        ),
        platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
    )

    assert completed.returncode == 0
    observed = json.loads(completed.stdout)
    assert observed == {
        "cpu": [2, 2],
        "memory": [address_space, address_space],
        "file": [1024 * 1024, 1024 * 1024],
    }


def test_cancellation_stops_worker_before_its_wall_time_budget() -> None:
    cancellation_event = threading.Event()
    timer = threading.Timer(0.2, cancellation_event.set)
    started = time.monotonic()
    timer.start()
    try:
        with bounded_process_cancellation(cancellation_event):
            completed = run_bounded_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                input_bytes=b"",
                timeout_seconds=20,
                environment=dict(os.environ),
                stdout_limit=4096,
                stderr_limit=4096,
            )
    finally:
        timer.cancel()

    assert completed.cancelled
    assert not completed.timed_out
    assert time.monotonic() - started < 3


def test_cancellation_before_spawn_returns_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation_event = threading.Event()
    cancellation_event.set()

    def unexpected_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected process spawn: {args!r} {kwargs!r}")

    monkeypatch.setattr(process_module.subprocess, "Popen", unexpected_spawn)
    completed = run_bounded_process(
        [sys.executable, "-c", "raise SystemExit(0)"],
        input_bytes=b"",
        timeout_seconds=1,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
        cancellation_event=cancellation_event,
    )

    assert completed.cancelled
    assert not completed.timed_out
    assert completed.returncode is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal grace is required")
def test_cancellation_allows_graceful_process_tree_termination(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    terminated = tmp_path / "terminated"
    cancellation_event = threading.Event()
    canceller = _cancel_after_marker(cancellation_event, ready)
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, signal, sys, time; "
                "ready=pathlib.Path(sys.argv[1]); stopped=pathlib.Path(sys.argv[2]); "
                "signal.signal(signal.SIGTERM, lambda *_: "
                "(stopped.write_text('term'), sys.exit(0))); "
                "ready.write_text('ready'); time.sleep(30)"
            ),
            str(ready),
            str(terminated),
        ],
        input_bytes=b"",
        timeout_seconds=20,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
        cancellation_event=cancellation_event,
    )
    canceller.join(timeout=1)

    assert completed.cancelled
    assert not completed.timed_out
    assert terminated.read_text(encoding="utf-8") == "term"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal escalation is required")
def test_cancellation_forces_a_process_that_ignores_sigterm(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    cancellation_event = threading.Event()
    canceller = _cancel_after_marker(cancellation_event, ready)
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
            ),
            str(ready),
        ],
        input_bytes=b"",
        timeout_seconds=20,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
        cancellation_event=cancellation_event,
    )
    canceller.join(timeout=1)

    assert completed.cancelled
    assert not completed.timed_out
    assert completed.returncode == -signal.SIGKILL


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-owned")
def test_cancellation_removes_the_complete_owned_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "processes.json"
    cancellation_event = threading.Event()
    canceller = _cancel_after_marker(cancellation_event, marker)
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import json, os, pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(30)']); "
                "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid])); "
                "time.sleep(30)"
            ),
            str(marker),
        ],
        input_bytes=b"",
        timeout_seconds=20,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
        cancellation_event=cancellation_event,
    )
    canceller.join(timeout=1)
    pids = json.loads(marker.read_text(encoding="utf-8"))

    assert completed.cancelled
    assert all(_wait_for_pid_exit(pid) for pid in pids)


def test_timeout_and_execution_failure_remain_distinct() -> None:
    timed_out = run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        input_bytes=b"",
        timeout_seconds=0.1,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
    )
    failed = run_bounded_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        input_bytes=b"",
        timeout_seconds=1,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
    )

    assert timed_out.timed_out and not timed_out.cancelled
    assert not failed.timed_out and not failed.cancelled
    assert failed.returncode == 7


@pytest.mark.skipif(
    os.name != "posix",
    reason="process-group descendant cleanup is exercised on POSIX",
)
def test_clean_worker_exit_drains_pipes_inherited_by_descendants() -> None:
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(30)'], "
                "stdout=sys.stdout, stderr=sys.stderr); "
                "print('worker complete', flush=True)"
            ),
        ],
        input_bytes=b"",
        timeout_seconds=0.5,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
    )

    assert completed.returncode == 0
    assert not completed.timed_out
    assert completed.stdout == b"worker complete\n"
    assert completed.stderr == b""


@pytest.mark.skipif(
    os.name != "posix",
    reason="detached process groups are exercised on POSIX",
)
def test_detached_descendant_with_inherited_pipe_fails_closed(
    tmp_path: Path,
) -> None:
    """Detached descendant keeps pipe open; result must fail closed (timed_out).

    The escaped descendant is killed in ``finally`` so no process is left
    running after the test, even if an assertion fails.
    """
    marker = tmp_path / "escaped.pid"
    script = tmp_path / "escape_worker.py"
    script.write_text(
        "import subprocess, sys\n"
        "p = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(5)'], "
        "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True)\n"
        "open(sys.argv[1], 'w').write(str(p.pid))\n"
        "print('worker complete', flush=True)\n",
        encoding="utf-8",
    )
    escaped_pid: int | None = None
    try:
        completed = run_bounded_process(
            [sys.executable, str(script), str(marker)],
            input_bytes=b"",
            timeout_seconds=2,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

        assert completed.returncode == 0
        assert completed.timed_out
    finally:
        if marker.exists():
            try:
                text = marker.read_text().strip()
                if text:
                    escaped_pid = int(text)
            except OSError:
                pass
        if escaped_pid is not None:
            with contextlib.suppress(OSError):
                os.kill(escaped_pid, 9)
