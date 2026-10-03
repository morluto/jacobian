"""Deterministic clocks at VF2's existing lease and worker boundaries."""

from types import SimpleNamespace
from typing import Any, Literal

import pytest

from jacobian import _execution, process
from jacobian._execution import (
    OperationPhaseLease,
    lease_operation_phases,
    require_execution_deadline,
)
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.math.graphs.isomorphism import _vf2_process as owner
from jacobian.process import BoundedProcessResult

TimeoutPath = Literal["reserve", "prelaunch", "worker", "delivery"]
VF2_PAYLOAD: dict[str, Any] = {
    "graph_a": {"vertex_count": 2, "directed": False, "edges": [[0, 1]]},
    "graph_b": {"vertex_count": 2, "directed": False, "edges": [[0, 1]]},
}


def patch_vf2_timeout(
    monkeypatch: pytest.MonkeyPatch, clock: list[float], path: TimeoutPath
) -> list[str]:
    """Reach a real owner timeout path without replacing the event-loop clock."""

    trace: list[str] = []
    local_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(_execution, "time", local_time)
    monkeypatch.setattr(owner, "time", local_time)

    def lease(
        wall_seconds: float, *, admitted_response_bytes: int, validation_work: int
    ) -> OperationPhaseLease:
        trace.append("lease")
        leased = lease_operation_phases(
            wall_seconds,
            admitted_response_bytes=admitted_response_bytes,
            validation_work=validation_work,
        )
        if path == "prelaunch":
            clock[0] = leased.backend_deadline + 0.001
        return leased

    def complete(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        trace.append("worker")
        assert path in {"worker", "delivery"}
        return BoundedProcessResult(
            returncode=0,
            stdout=encode_worker_result_frame(
                {"ok": True, "mapping": [[0, 0], [1, 1]]}
            ),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=path == "worker",
        )

    def checkpoint(deadline: float) -> None:
        trace.append("delivery")
        clock[0] = deadline + 0.001
        require_execution_deadline(deadline)

    monkeypatch.setattr(owner, "lease_operation_phases", lease)
    monkeypatch.setattr(process, "run_bounded_process", complete)
    monkeypatch.setattr(owner, "require_execution_deadline", checkpoint)
    return trace
