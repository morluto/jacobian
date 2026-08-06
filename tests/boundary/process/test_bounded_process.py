from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import time

import pytest

from jacobian.bounded_process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    bounded_process_cancellation,
    run_bounded_process,
)


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
def test_detached_descendant_with_inherited_pipe_fails_closed() -> None:
    completed = run_bounded_process(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(5)'], "
                "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True); "
                "print('worker complete', flush=True)"
            ),
        ],
        input_bytes=b"",
        timeout_seconds=2,
        environment=dict(os.environ),
        stdout_limit=4096,
        stderr_limit=4096,
    )

    assert completed.returncode == 0
    assert completed.timed_out
