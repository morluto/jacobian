from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pytest
import z3  # type: ignore[import-untyped]

from jacobian.math.graphs import _independence_z3
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
)
from jacobian.math.graphs.optimization import (
    _chromatic_number,
    _exact_search,
    _finite_optimization,
    _invariants,
)
from jacobian.math.graphs.optimization._coloring_models import (
    ChromaticGraph,
    GraphChromaticNumberRequest,
)
from jacobian.math.graphs.optimization._models import (
    GraphDominationMinimumOutput,
    GraphMinimumMaximalMatchingOutput,
    GraphOptimizationBudget,
    GraphOptimizationRequest,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _graph() -> ChromaticGraph:
    return ChromaticGraph(vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c")))


def _expired(monkeypatch: pytest.MonkeyPatch, entry_module: Any) -> None:
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(entry_module.time, "monotonic", lambda: next(clock, 2.0))


def test_clique_budget_starts_before_incumbent_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _expired(monkeypatch, _invariants)
    monkeypatch.setattr(
        nx.approximation,
        "max_clique",
        lambda _graph: (_ for _ in ()).throw(
            AssertionError("uninterruptible clique seed must not run")
        ),
    )
    request = GraphOptimizationRequest(
        graph=_graph(), resource_budget=GraphOptimizationBudget(wall_seconds=1)
    )

    result = _invariants._clique_execute_kernel(request)

    assert result.status == "UNKNOWN"
    assert result.termination_reason == "WALL_TIME"
    assert result.tested == ()


def test_chromatic_budget_starts_before_graph_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _expired(monkeypatch, _chromatic_number)
    monkeypatch.setattr(
        nx.coloring,
        "greedy_color",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("uninterruptible coloring seed must not run")
        ),
    )
    request = GraphChromaticNumberRequest.model_validate(
        {"graph": _graph().model_dump(), "resource_budget": {"wall_seconds": 1}}
    )

    result = _chromatic_number._search_chromatic_number_kernel(request)

    assert result.status == "UNKNOWN"
    assert result.solver_status == "UNKNOWN"
    assert "wall-clock budget expired" in result.detail
    assert result.tested == ()


def test_chromatic_worker_projection_is_bound_to_the_submitted_vertices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GraphChromaticNumberRequest.model_validate(
        {"graph": _graph().model_dump(), "resource_budget": {"wall_seconds": 3}}
    )
    expected = _chromatic_number._search_chromatic_number_kernel(request)

    monkeypatch.setattr(
        _chromatic_number,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=0,
            stdout=json.dumps(
                expected.model_dump(mode="json", exclude={"vertices"}),
                ensure_ascii=False,
            ).encode("utf-8"),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    )

    assert _chromatic_number._search_chromatic_number(request) == expected


@pytest.mark.parametrize(
    "operation_id",
    [
        "graph.domination.minimum.compute",
        "graph.matching.maximal.minimum.compute",
    ],
)
def test_finite_searches_do_not_reset_the_operation_timer(
    monkeypatch: pytest.MonkeyPatch, operation_id: str
) -> None:
    _expired(monkeypatch, _finite_optimization)
    request = GraphOptimizationRequest(
        graph=_graph(), resource_budget=GraphOptimizationBudget(wall_seconds=1)
    )

    result = cast(
        GraphDominationMinimumOutput | GraphMinimumMaximalMatchingOutput,
        _finite_optimization._run_worker_kernel(operation_id, request),
    )

    assert result.status == "UNKNOWN"
    assert result.termination_reason == "WALL_TIME"
    assert result.tested == ()


def test_graph_optimization_worker_binds_encoding_and_solving_to_one_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GraphOptimizationRequest(
        graph=_graph(), resource_budget=GraphOptimizationBudget(wall_seconds=3)
    )
    expected = _finite_optimization._run_worker_kernel(
        "graph.domination.minimum.compute", request
    )
    recorded: dict[str, object] = {}

    def complete_worker(*args: object, **kwargs: object) -> BoundedProcessResult:
        recorded["args"] = args
        recorded.update(kwargs)
        return BoundedProcessResult(
            returncode=0,
            stdout=json.dumps(expected.model_dump(mode="json")).encode("utf-8"),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(_finite_optimization, "run_bounded_process", complete_worker)

    result = _finite_optimization.DOMINATION_MINIMUM_OPERATION.run(request)

    assert result == expected
    timeout_seconds = recorded["timeout_seconds"]
    assert isinstance(timeout_seconds, float)
    assert 0 < timeout_seconds <= 3
    assert Path(str(recorded["cwd"])).name.startswith("jacobian-graph-optimization-")
    limits = recorded["resource_limits"]
    assert isinstance(limits, ProcessResourceLimits)
    assert limits.cpu_seconds == 3
    assert limits.address_space_bytes == 1_536 * 1024 * 1024
    assert limits.file_size_bytes == 1_024 * 1_024


def test_graph_optimization_worker_failure_cannot_claim_an_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _finite_optimization,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = _finite_optimization.DOMINATION_MINIMUM_OPERATION.run(
        GraphOptimizationRequest(graph=_graph())
    )

    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.termination_reason == "SOLVER_UNKNOWN"


def test_threshold_solver_does_not_start_or_finish_after_encoding_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Solver:
        def __init__(self) -> None:
            self.set_called = False
            self.check_called = False

        def set(self, **_settings: int) -> None:
            self.set_called = True

        def check(self) -> object:
            self.check_called = True
            return z3.sat

    budget = GraphOptimizationBudget(wall_seconds=1)
    solver = Solver()
    monkeypatch.setattr(_exact_search, "_remaining_ms", lambda *_args: 0)

    assert (
        _exact_search._check_after_encoding(solver, started=0.0, budget=budget)
        == z3.unknown
    )
    assert not solver.set_called
    assert not solver.check_called

    elapsed = iter((1_000, 0))
    monkeypatch.setattr(_exact_search, "_remaining_ms", lambda *_args: next(elapsed))
    assert (
        _exact_search._check_after_encoding(solver, started=0.0, budget=budget)
        == z3.unknown
    )
    assert solver.set_called
    assert solver.check_called


def test_independence_does_not_enter_z3_after_seed_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(
        "jacobian.math.graphs._independence_z3.time.monotonic", lambda: next(clock)
    )
    request = IndependenceNumberRequest(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
        ),
        resource_budget=IndependenceNumberBudget(wall_seconds=1),
    )

    result = _independence_z3._solve_independence_number_values_kernel(
        request.graph, request.resource_budget
    )

    assert result.status == "UNKNOWN"
    assert result.termination_reason == "WALL_TIME"
    assert result.witness_vertices == ("a",)
