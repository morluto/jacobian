"""Exercise a production worker timeout without waiting for a solver deadline."""

import io
import json
import sys
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
import z3

from jacobian import _execution as execution
from jacobian.math.graphs.optimization import _budget as budget
from jacobian.math.graphs.optimization import _chromatic_bipartition as kernel
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization import _chromatic_bipartition_worker as worker
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
)


def patch_chromatic_clock(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    """Control only the operation clocks; preserve the live SDK/event-loop clock."""
    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    for owner in (
        execution,
        budget,
        kernel,
        process_owner,
        worker,
        sys.modules[__name__],
    ):
        monkeypatch.setattr(owner, "time", fake_time)


def chromatic_timeout_request(wall_seconds: int = 5) -> ChromaticBipartitionRequest:
    """Require one nontrivial coloring decision in the first singleton split."""
    return ChromaticBipartitionRequest.model_validate(
        {
            "graph": {
                "vertices": list("abcdef"),
                "edges": [["b", "c"], ["c", "d"], ["d", "e"], ["e", "f"], ["b", "f"]],
            },
            "s": 1,
            "t": 1,
            "resource_budget": {"wall_seconds": wall_seconds},
        }
    )


def chromatic_timeout_worker_output(
    monkeypatch: pytest.MonkeyPatch, input_bytes: bytes
) -> tuple[bytes, list[int]]:
    """Run the real child entry point on its parent's actual admitted payload."""
    wall_seconds = json.loads(input_bytes)["resource_budget"]["wall_seconds"]
    solver_calls = 0
    solver_timeouts: list[int] = []

    class TimeoutSolver:
        def set(self, *, timeout: int) -> None:
            assert 1 <= timeout <= wall_seconds * 1000
            solver_timeouts.append(timeout)

        def add(self, *clauses: object) -> None:
            assert clauses

        def check(self) -> object:
            nonlocal solver_calls
            solver_calls += 1
            return z3.unknown

    started = time.monotonic()
    output = io.StringIO()
    with monkeypatch.context() as worker_patch:
        patch_chromatic_clock(worker_patch, [started])
        worker_patch.setattr(z3, "Solver", TimeoutSolver)
        worker_patch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(input_bytes)))
        with redirect_stdout(output):
            assert worker.main() == 0
    assert solver_calls == len(solver_timeouts)
    return output.getvalue().encode(), solver_timeouts


def chromatic_worker_timeout(
    monkeypatch: pytest.MonkeyPatch,
    wall_seconds: int = 5,
    *,
    expired_parent_deadline: bool = False,
) -> tuple[ChromaticBipartitionRequest, bytes]:
    """Let the actual kernel classify one unsettled C5 coloring decision."""
    request = chromatic_timeout_request(wall_seconds)
    remaining = -1 if expired_parent_deadline else wall_seconds
    payload = {
        "_deadline": time.monotonic() + remaining,
        **request.model_dump(mode="json"),
    }
    output, solver_timeouts = chromatic_timeout_worker_output(
        monkeypatch, json.dumps(payload).encode()
    )
    assert len(solver_timeouts) == (0 if expired_parent_deadline else 1)
    return request, output
