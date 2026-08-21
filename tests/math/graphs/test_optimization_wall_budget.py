from __future__ import annotations

import networkx as nx
import pytest

from jacobian.math.graphs import _independence_z3
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceNumberRequest,
)
from jacobian.math.graphs.optimization import (
    _chromatic_number,
    _finite_optimization,
    _invariants,
)
from jacobian.math.graphs.optimization._coloring_models import (
    ChromaticGraph,
    GraphChromaticNumberRequest,
)
from jacobian.math.graphs.optimization._models import (
    GraphOptimizationBudget,
    GraphOptimizationRequest,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph() -> ChromaticGraph:
    return ChromaticGraph(vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c")))


def _expired(monkeypatch: pytest.MonkeyPatch, entry_module: object) -> None:
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

    result = _invariants._clique_execute(request)

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

    result = _chromatic_number._search_chromatic_number(request)

    assert result.status == "UNKNOWN"
    assert result.solver_status == "UNKNOWN"
    assert "wall-clock budget expired" in result.detail
    assert result.tested == ()


@pytest.mark.parametrize(
    ("operation", "seed_name"),
    [
        (_finite_optimization.DOMINATION_MINIMUM_OPERATION, "dominating_set"),
        (
            _finite_optimization.MINIMUM_MAXIMAL_MATCHING_OPERATION,
            "maximal_matching",
        ),
    ],
)
def test_finite_searches_do_not_reset_the_operation_timer(
    monkeypatch: pytest.MonkeyPatch, operation: object, seed_name: str
) -> None:
    _expired(monkeypatch, _finite_optimization)
    monkeypatch.setattr(
        nx,
        seed_name,
        lambda _graph: (_ for _ in ()).throw(
            AssertionError(f"uninterruptible {seed_name} seed must not run")
        ),
    )
    request = GraphOptimizationRequest(
        graph=_graph(), resource_budget=GraphOptimizationBudget(wall_seconds=1)
    )

    result = operation.run(request)

    assert result.status == "UNKNOWN"
    assert result.termination_reason == "WALL_TIME"
    assert result.tested == ()


def test_independence_does_not_enter_z3_after_seed_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(_independence_z3.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        nx.approximation,
        "maximum_independent_set",
        lambda _graph: (_ for _ in ()).throw(
            AssertionError("uninterruptible independence seed must not run")
        ),
    )

    class ForbiddenOptimizer:
        def __init__(self) -> None:
            raise AssertionError("Z3 must not start after the operation deadline")

    monkeypatch.setattr(_independence_z3.z3, "Optimize", ForbiddenOptimizer)
    request = IndependenceNumberRequest(
        graph=SimpleUndirectedGraph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"))
        ),
        resource_budget=IndependenceNumberBudget(wall_seconds=1),
    )

    result = _independence_z3.solve_independence_number(request)

    assert result.status == "UNKNOWN"
    assert result.termination_reason == "WALL_TIME"
    assert result.witness_vertices == ("a",)
