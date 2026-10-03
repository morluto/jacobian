"""Exercise a production worker timeout without waiting for a solver deadline."""

import io
import json
import sys
import time
from contextlib import redirect_stdout

import pytest
import z3

from jacobian.math.graphs.optimization import _chromatic_bipartition_worker as worker
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
)


def chromatic_worker_timeout(
    monkeypatch: pytest.MonkeyPatch,
    wall_seconds: int = 5,
    *,
    expired_parent_deadline: bool = False,
) -> tuple[ChromaticBipartitionRequest, bytes]:
    """Let the actual kernel classify one unsettled C5 coloring decision."""
    request = ChromaticBipartitionRequest.model_validate(
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
    solver_calls = 0

    class TimeoutSolver:
        def set(self, *, timeout: int) -> None:
            assert 1 <= timeout <= wall_seconds * 1000

        def add(self, *clauses: object) -> None:
            assert clauses

        def check(self) -> object:
            nonlocal solver_calls
            solver_calls += 1
            return z3.unknown

    remaining = -1 if expired_parent_deadline else wall_seconds
    started = time.monotonic()
    payload = {"_deadline": started + remaining, **request.model_dump(mode="json")}
    output = io.StringIO()
    with monkeypatch.context() as worker_patch:
        worker_patch.setattr(time, "monotonic", lambda: started)
        worker_patch.setattr(z3, "Solver", TimeoutSolver)
        worker_patch.setattr(
            sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))
        )
        with redirect_stdout(output):
            assert worker.main() == 0
    assert solver_calls == (0 if expired_parent_deadline else 1)
    return request, output.getvalue().encode()
