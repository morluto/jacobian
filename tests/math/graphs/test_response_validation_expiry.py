"""A real valid worker result expiring during parsing is not malformed output."""

import time
from threading import Event
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
)
from jacobian.math.graphs.optimization import _chromatic_number, _invariants
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberRequest,
)
from jacobian.math.graphs.optimization._models import GraphOptimizationRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize("kind", ["chromatic", "clique"])
@pytest.mark.parametrize("cancel", [False, True])
def test_control_expiry_after_real_response_validation(
    monkeypatch: pytest.MonkeyPatch, kind: str, cancel: bool
) -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    if kind == "chromatic":
        module: Any = _chromatic_number
        request: Any = GraphChromaticNumberRequest.model_validate(
            {"graph": graph, "resource_budget": {"wall_seconds": 1}}
        )
        expected = module._search_chromatic_number_kernel(request)
        execute = module._search_chromatic_number
    else:
        module = _invariants
        request = GraphOptimizationRequest.model_validate(
            {"graph": graph, "resource_budget": {"wall_seconds": 1}}
        )
        expected = module._clique_execute_kernel(request)
        execute = module._clique_execute
    response = BoundedProcessResult(
        returncode=0,
        stdout=expected.model_dump_json().encode(),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
    )
    monkeypatch.setattr(module, "run_bounded_process", lambda *a, **_kwargs: response)
    now = [0.0]
    event = Event()
    original = type(expected).model_validate

    def delayed(cls: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(value, *args, **kwargs)
        if cancel:
            event.set()
        else:
            now[0] = 2.0
        return result

    monkeypatch.setattr(type(expected), "model_validate", classmethod(delayed))
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    with request_cancellation(event):
        if cancel:
            with pytest.raises(OperationExecutionCancelledError):
                execute(request)
        else:
            with pytest.raises(OperationExecutionTimeoutError):
                execute(request)


def test_augmentation_noop_expiry_has_a_modeled_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.graphs.triangle_free_diameter_augmentation import (
        _augmentation_z3 as module,
    )
    from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
        TriangleFreeDiameterAugmentationBudget,
    )

    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    now = [0.0]
    original = module._diameter

    def delayed(value: SimpleUndirectedGraph) -> int | None:
        result = original(value)
        now[0] = 2.0
        return result

    monkeypatch.setattr(module, "_diameter", delayed)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    result = module.solve_triangle_free_diameter_augmentation_values(
        graph, 1, TriangleFreeDiameterAugmentationBudget(wall_seconds=1)
    )
    assert result.status == "SOLVER_BUDGET_EXCEEDED"
    assert "no-op presolve" in result.detail


@pytest.mark.parametrize("cancel", [False, True])
def test_hypergraph_response_validation_control_outcome(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3 as module,
    )
    from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
        FiniteHypergraph,
        HypergraphIndependenceBudget,
        HypergraphIndependenceResult,
    )

    source = FiniteHypergraph(vertices=("a", "b", "c"), edges=(("edge", ("a", "b")),))
    budget = HypergraphIndependenceBudget(wall_seconds=1)
    expected = module._solve_independence_number_kernel(source, budget)
    monkeypatch.setattr(
        module,
        "_run_independence_worker",
        lambda *a, **_kwargs: expected.model_dump(
            mode="json", exclude={"hypergraph", "resource_budget"}
        ),
    )
    now = [0.0]
    event = Event()
    original = HypergraphIndependenceResult.model_validate

    def delayed(cls: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(value, *args, **kwargs)
        if cancel:
            event.set()
        else:
            now[0] = 2.0
        return result

    monkeypatch.setattr(
        HypergraphIndependenceResult, "model_validate", classmethod(delayed)
    )
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    with request_cancellation(event):
        if cancel:
            with pytest.raises(OperationExecutionCancelledError):
                module.solve_independence_number(source, budget)
        else:
            with pytest.raises(OperationExecutionTimeoutError):
                module.solve_independence_number(source, budget)


def test_hypergraph_worker_timeout_is_not_solver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3 as module,
    )
    from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
        FiniteHypergraph,
        HypergraphIndependenceBudget,
    )

    source = FiniteHypergraph(vertices=("a", "b", "c"), edges=(("edge", ("a", "b")),))
    monkeypatch.setattr(
        module,
        "run_bounded_process",
        lambda *a, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )
    with pytest.raises(OperationExecutionTimeoutError):
        module.solve_independence_number(
            source, HypergraphIndependenceBudget(wall_seconds=1)
        )


@pytest.mark.parametrize("cancel", [False, True])
def test_expiry_during_finite_graph_witness_check_precedes_invalid_witness(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    from jacobian.math.graphs.optimization import _finite_optimization as module

    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    request = GraphOptimizationRequest.model_validate(
        {"graph": graph, "resource_budget": {"wall_seconds": 1}}
    )
    operation = module.DOMINATION_MINIMUM_OPERATION
    expected = module._run_worker_kernel(operation.operation_id, request)
    monkeypatch.setattr(
        module,
        "run_bounded_process",
        lambda *a, **_kwargs: BoundedProcessResult(
            returncode=0,
            stdout=expected.model_dump_json().encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    )
    now = [0.0]
    event = Event()

    def delayed(*args: object) -> bool:
        if cancel:
            event.set()
        else:
            now[0] = 2.0
        return False

    monkeypatch.setattr(module, "_valid_witness", delayed)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    with request_cancellation(event):
        if cancel:
            with pytest.raises(OperationExecutionCancelledError):
                operation.run(request)
        else:
            with pytest.raises(OperationExecutionTimeoutError):
                operation.run(request)
