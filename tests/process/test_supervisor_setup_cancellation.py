"""Cancellation during real worker setup retains supervisor precedence."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace

import pytest

from jacobian import process
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.combinatorics._counting_process import (
    _COUNTING_WORKER,
    _decode_count_result,
)
from jacobian.process import ProcessPlatformTools, ProcessResourceLimits


@pytest.mark.parametrize("checked", [False, True], ids=["raw", "checked"])
@pytest.mark.parametrize(
    ("cancelled", "expired", "child_exited"),
    [
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (False, False, True),
    ],
    ids=["cancel-and-expiry", "expiry-control", "cancel-after-exit", "success-control"],
)
def test_setup_cancellation_is_checked_before_expiry_or_child_exit(
    monkeypatch: pytest.MonkeyPatch,
    checked: bool,
    cancelled: bool,
    expired: bool,
    child_exited: bool,
) -> None:
    started = time.monotonic()
    clock = [started]
    cancellation = Event()
    children: list[subprocess.Popen[bytes]] = []
    apply_limits = process._apply_post_start_limits

    def finish_setup(
        child: subprocess.Popen[bytes],
        limits: ProcessResourceLimits | None,
        applied_before_exec: bool,
        tools: ProcessPlatformTools | None,
    ) -> None:
        apply_limits(child, limits, applied_before_exec, tools)
        children.append(child)
        if child_exited:
            # Finish the real first-party worker before monitoring starts.
            assert child.wait(timeout=5) == 0
        if cancelled:
            cancellation.set()
        if expired:
            # Setup exhausts execution while retaining finite cleanup time.
            clock[0] = started + 29.9

    monkeypatch.setattr(process, "_apply_post_start_limits", finish_setup)
    monkeypatch.setattr(process, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    command = [sys.executable, str(_COUNTING_WORKER)]
    if checked:
        if cancelled or expired:
            expected = (
                OperationExecutionCancelledError
                if cancelled
                else OperationExecutionTimeoutError
            )
            with pytest.raises(expected):
                process.run_checked_worker_process(
                    command,
                    input_bytes=b'{"op":"comb","n":4,"k":2}',
                    timeout_seconds=30,
                    environment=dict(os.environ),
                    stdout_limit=4096,
                    stderr_limit=4096,
                    cancellation_event=cancellation,
                    decode_result=_decode_count_result,
                )
        else:
            assert (
                process.run_checked_worker_process(
                    command,
                    input_bytes=b'{"op":"comb","n":4,"k":2}',
                    timeout_seconds=30,
                    environment=dict(os.environ),
                    stdout_limit=4096,
                    stderr_limit=4096,
                    cancellation_event=cancellation,
                    decode_result=_decode_count_result,
                )
                == "6"
            )
    else:
        result = process.run_bounded_process(
            command,
            input_bytes=b'{"op":"comb","n":4,"k":2}',
            timeout_seconds=30,
            environment=dict(os.environ),
            stdout_limit=4096,
            stderr_limit=4096,
            cancellation_event=cancellation,
        )
        assert result.cancelled is cancelled
        assert result.timed_out is (expired and not cancelled)
        if not cancelled and not expired:
            assert result.stdout == b'{"kind":"result","result":"6"}\n'

    assert len(children) == 1
    assert children[0].poll() is not None
