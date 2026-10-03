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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Never

import pytest

from jacobian._execution import (
    bind_request_deadline,
    request_cancellation,
    request_execution,
)
from jacobian.process import (
    BoundedProcessResult,
    ProcessPlatformTools,
    ProcessResourceLimits,
    bounded_process_cancellation,
    run_bounded_process,
)


def test_expired_input_spooling_does_not_launch_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr("jacobian.process.time.monotonic", lambda: next(monotonic))

    def fail_to_launch(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("expired request must not launch a worker")

    monkeypatch.setattr("jacobian.process.subprocess.Popen", fail_to_launch)

    completed = run_bounded_process(
        [sys.executable, "-c", "raise SystemExit(0)"],
        input_bytes=b"input",
        timeout_seconds=1,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
    )

    assert completed.timed_out
    assert not completed.cancelled
    assert completed.returncode is None


def test_expired_request_envelope_does_not_launch_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_launch(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("expired request must not launch a worker")

    monkeypatch.setattr("jacobian.process.subprocess.Popen", fail_to_launch)

    with request_execution(0.0):
        bind_request_deadline(0.0)
        completed = run_bounded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            input_bytes=b"input",
            timeout_seconds=30,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

    assert completed.timed_out
    assert not completed.cancelled
    assert completed.returncode is None


def test_request_envelope_caps_worker_wall_time() -> None:
    started = time.monotonic()
    with request_execution(started):
        bind_request_deadline(started + 0.2)
        completed = run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_bytes=b"",
            timeout_seconds=30,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

    assert completed.timed_out
    assert not completed.cancelled
    assert completed.returncode is not None
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("envelope_only", [False, True], ids=["legacy", "envelope"])
def test_cancelled_request_does_not_launch_a_worker(
    monkeypatch: pytest.MonkeyPatch,
    envelope_only: bool,
) -> None:
    def fail_to_launch(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("cancelled request must not launch a worker")

    monkeypatch.setattr("jacobian.process.subprocess.Popen", fail_to_launch)
    cancellation_event = threading.Event()
    cancellation_event.set()

    cancellation_scope = (
        request_execution(time.monotonic(), cancellation_signal=cancellation_event)
        if envelope_only
        else request_cancellation(cancellation_event)
    )
    with cancellation_scope:
        completed = run_bounded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            input_bytes=b"input",
            timeout_seconds=30,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

    assert completed.cancelled
    assert not completed.timed_out
    assert completed.returncode is None
    assert completed.stdout == completed.stderr == b""


@pytest.mark.parametrize(
    ("explicit", "legacy", "envelope", "cancelled"),
    [
        (None, None, None, False),
        (None, None, False, False),
        (None, False, True, False),
        (None, True, False, True),
        (False, None, True, False),
        (False, True, True, False),
        (True, False, False, True),
    ],
)
def test_raw_worker_cancellation_signal_precedence(
    explicit: bool | None,
    legacy: bool | None,
    envelope: bool | None,
    cancelled: bool,
) -> None:
    class FalseyEvent(threading.Event):
        def __bool__(self) -> bool:
            return False

    def event(value: bool | None) -> threading.Event | None:
        if value is None:
            return None
        # The signal protocol requires only is_set(), not truthiness.
        signal = FalseyEvent()
        if value:
            signal.set()
        return signal

    legacy_event = event(legacy)
    with (
        request_execution(time.monotonic(), cancellation_signal=event(envelope)),
        request_cancellation(legacy_event)
        if legacy_event is not None
        else contextlib.nullcontext(),
    ):
        completed = run_bounded_process(
            [sys.executable, "-c", "print('completed')"],
            input_bytes=b"",
            timeout_seconds=5,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
            cancellation_event=event(explicit),
        )

    assert completed.cancelled is cancelled
    assert not completed.timed_out
    assert completed.returncode == (None if cancelled else 0)
    assert completed.stdout == (b"" if cancelled else b"completed\n")
    assert completed.stderr == b""


@pytest.mark.skipif(os.name != "posix", reason="process-tree assertion is POSIX")
def test_envelope_cancellation_stops_running_worker_and_descendant(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ready.json"
    script = (
        "import json, os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "parent = os.getpid()\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    marker = pathlib.Path(sys.argv[1])\n"
        "    temporary = marker.with_suffix('.tmp')\n"
        "    temporary.write_text(json.dumps([parent, os.getpid()]))\n"
        "    temporary.replace(marker)\n"
        "time.sleep(30)\n"
    )
    cancellation = threading.Event()

    def run_worker() -> BoundedProcessResult:
        with request_execution(time.monotonic(), cancellation_signal=cancellation):
            return run_bounded_process(
                [sys.executable, "-c", script, str(marker)],
                input_bytes=b"",
                timeout_seconds=20,
                environment=dict(os.environ),
                stdout_limit=4096,
                stderr_limit=4096,
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run_worker)
        try:
            ready_deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < ready_deadline:
                time.sleep(0.01)
            assert marker.exists(), "worker descendant did not become ready"
            pids = json.loads(marker.read_text())
            for pid in pids:
                os.kill(pid, 0)

            cancellation.set()
            # Both processes sleep for 30 seconds and the supervisor allows 20.
            # Returning within 5 seconds proves cancellation, not natural exit.
            completed = pending.result(timeout=5)
            assert completed.cancelled
            assert not completed.timed_out
            assert completed.returncode == -signal.SIGKILL

            reap_deadline = time.monotonic() + 5
            remaining = set(pids)
            while remaining and time.monotonic() < reap_deadline:
                for pid in tuple(remaining):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        remaining.remove(pid)
                if remaining:
                    time.sleep(0.01)
            assert not remaining, f"cancelled process tree survived: {remaining}"
        finally:
            if marker.exists():
                parent, _child = json.loads(marker.read_text())
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(parent, signal.SIGKILL)


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


def test_missing_process_output_pipe_fails_before_reader_threads_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )

    class MissingStdout:
        pid = child.pid
        stdin = None
        stdout = None
        stderr = child.stderr
        returncode: int | None = None

        def poll(self) -> int | None:
            return child.poll()

        def kill(self) -> None:
            child.kill()

        def wait(self, timeout: float | None = None) -> int:
            return child.wait(timeout=timeout)

    monkeypatch.setattr(
        "jacobian.process.subprocess.Popen", lambda *_args, **_kwargs: MissingStdout()
    )

    with pytest.raises(RuntimeError, match="stdout pipe was not created"):
        run_bounded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            input_bytes=b"",
            timeout_seconds=1,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

    assert child.poll() is not None


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
        "[sys.executable, '-c', 'import time; time.sleep(2)'], "
        "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True)\n"
        "open(sys.argv[1], 'w').write(str(p.pid))\n"
        "print('worker complete', flush=True)\n",
        encoding="utf-8",
    )
    escaped_pid: int | None = None
    started = time.monotonic()
    try:
        completed = run_bounded_process(
            [sys.executable, str(script), str(marker)],
            input_bytes=b"",
            timeout_seconds=0.5,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
        )

        assert completed.returncode == 0
        assert completed.timed_out
        assert time.monotonic() - started < 1.5
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
