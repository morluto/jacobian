"""Backend calls are interruptible, including square-free decomposition."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_cancellation,
    request_execution,
)
from jacobian.math.number_theory.galois import _factor_process
from jacobian.math.number_theory.galois.operations import galois_factor


@pytest.mark.parametrize("phase", ["gf_sqf_list", "gf_berlekamp"])
@pytest.mark.parametrize("cancel", [False, True])
def test_request_stops_and_reaps_a_blocked_backend_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, cancel: bool
) -> None:
    marker = tmp_path / "backend-started"
    worker = tmp_path / "blocked_worker.py"
    worker.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "import sympy.polys.galoistools as backend\n"
        "from jacobian.math.number_theory.galois._factor_worker import main\n"
        "def blocked(*args, **kwargs):\n"
        f"    Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        f"backend.{phase} = blocked\n"
        "raise SystemExit(main())\n"
    )
    monkeypatch.setattr(_factor_process, "_FACTOR_WORKER", worker)
    cancellation = Event()
    stop = Event()

    def cancel_after_backend_starts() -> None:
        while not stop.wait(0.01):
            if marker.exists():
                cancellation.set()
                return

    thread = Thread(target=cancel_after_backend_starts) if cancel else None
    if thread is not None:
        thread.start()
    started = time.monotonic()
    error = (
        OperationExecutionCancelledError if cancel else OperationExecutionTimeoutError
    )
    try:
        with request_execution(started), request_cancellation(cancellation):
            # The caller's earlier deadline must win over the owner's 60s cap.
            bind_request_deadline(started + 5)
            with pytest.raises(error):
                galois_factor(5, (1, 0, 1))
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1)
    assert marker.exists(), "the stop must occur inside the backend call"
    assert time.monotonic() - started < 8
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)


def test_cancellation_during_directory_setup_does_not_start_a_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _factor_process.TemporaryDirectory
    cancellation = Event()

    @contextmanager
    def expired_directory(*args: Any, **kwargs: Any) -> Iterator[str]:
        with original(*args, **kwargs) as directory:
            cancellation.set()
            yield directory

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an expired request must not start the factorization worker")

    monkeypatch.setattr(_factor_process, "TemporaryDirectory", expired_directory)
    monkeypatch.setattr(_factor_process, "run_bounded_process", unexpected)
    with (
        request_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError),
    ):
        galois_factor(5, (1, 0, 1))
