from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Never

import pytest

from jacobian.process import (
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
