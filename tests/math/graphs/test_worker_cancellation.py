"""Stopped accelerators and exact workers preserve operation control outcomes."""

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.graphs.isomorphism import _vf2_process
from jacobian.math.graphs.isomorphism._models import SimpleGraph
from jacobian.math.graphs.optimization import _maximum_cut_process
from jacobian.math.graphs.optimization._maximum_cut import GraphMaximumCutRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.logic import _unsat_core
from jacobian.math.logic._smt import SmtLogic
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize("kind", ["maximum_cut", "isomorphism", "unsat_core"])
def test_cancelled_worker_does_not_fall_back_or_return_unknown(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import jacobian.process as process

    completed = BoundedProcessResult(
        returncode=None,
        stdout=b"",
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
        cancelled=True,
    )
    monkeypatch.setattr(process, "run_bounded_process", lambda *a, **kw: completed)
    monkeypatch.setattr(
        _maximum_cut_process, "run_bounded_process", lambda *a, **kw: completed
    )
    monkeypatch.setattr(_unsat_core, "run_bounded_process", lambda *a, **kw: completed)
    with pytest.raises(OperationExecutionCancelledError):
        if kind == "maximum_cut":
            _maximum_cut_process.compute_maximum_cut_isolated(
                GraphMaximumCutRequest(
                    graph=SimpleUndirectedGraph(
                        vertices=("a", "b"), edges=(("a", "b"),)
                    )
                )
            )
        elif kind == "isomorphism":
            graph = SimpleGraph(vertex_count=2, edges=((0, 1),))
            _vf2_process._vertex_mapping(graph, graph)
        else:
            request = _unsat_core.SmtUnsatCoreRequest(
                logic=SmtLogic("QF_LIA"),
                smtlib="(set-logic QF_LIA) (assert false) (check-sat)",
                timeout_ms=1000,
                rlimit=100000,
            )
            _unsat_core.compute_smt_unsat_core(request)


def test_vf2_timeout_has_operation_execution_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.process as process

    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *a, **kw: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )
    graph = SimpleGraph(vertex_count=2, edges=((0, 1),))
    with pytest.raises(OperationExecutionTimeoutError) as error:
        _vf2_process._vertex_mapping(graph, graph)
    assert error.value.stage == "operation_execution"
