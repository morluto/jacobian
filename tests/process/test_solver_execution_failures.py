"""Failed worker execution must not establish a mathematical outcome (#3560-3562)."""

import importlib
from typing import Any, Never

import pytest

from jacobian._execution import (
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.catalog.catalog import Catalog
from jacobian.process import BoundedProcessResult, encode_worker_result_frame

OWNERS = [
    ("sat.solve", "jacobian.math.logic._sat"),
    ("smt.solve", "jacobian.math.logic._smt"),
    ("smt.unsat_core", "jacobian.math.logic._unsat_core"),
    *[
        (name, "jacobian.math.graphs.optimization._finite_optimization")
        for name in (
            "graph.domination.minimum.compute",
            "graph.induced_bipartite.maximum.compute",
            "graph.induced_forest.maximum.compute",
            "graph.induced_tree.maximum.compute",
            "graph.matching.maximal.minimum.compute",
        )
    ],
    (
        "graph.invariant.clique_number.compute",
        "jacobian.math.graphs.optimization._invariants",
    ),
    (
        "graph.invariant.chromatic_number.compute",
        "jacobian.math.graphs.optimization._chromatic_number",
    ),
    (
        "graph.invariant.independence_number.compute",
        "jacobian.math.graphs._independence_z3",
    ),
    (
        "hypergraph.independence_number.compute",
        "jacobian.math.combinatorics.finite_structures.hypergraphs._independence_z3",
    ),
]


@pytest.mark.parametrize(("operation_id", "module"), OWNERS)
def test_startup_failure_raises(
    monkeypatch: pytest.MonkeyPatch, operation_id: str, module: str
) -> None:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    request = operation.request_type.model_validate(operation.examples[0].input)

    def fail(*args: object, **kwargs: object) -> Never:
        raise OSError("private startup marker")

    monkeypatch.setattr(importlib.import_module(module), "run_bounded_process", fail)
    with pytest.raises(OperationBackendError) as caught:
        operation.run(request)
    assert caught.value.reason == "startup"
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize(("operation_id", "module"), OWNERS)
@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"returncode": 2}, OperationBackendError),
        ({"stdout_exceeded": True}, OperationResourceExhaustedError),
        ({"stderr_exceeded": True}, OperationResourceExhaustedError),
        ({"stdout": b"not JSON"}, OperationBackendError),
        ({"stdout": b'{"invalid":"result"}'}, OperationBackendError),
        ({"cancelled": True, "timed_out": True}, OperationExecutionCancelledError),
        ({"timed_out": True, "stdout_exceeded": True}, OperationExecutionTimeoutError),
        (
            {
                "stdout": b'{"kind":"execution_error","stage":"operation_execution","resource":"memory"}'
            },
            OperationResourceExhaustedError,
        ),
        (
            {
                "stdout": b'{"kind":"execution_error","stage":"operation_execution","resource":"forged"}'
            },
            OperationBackendError,
        ),
    ],
)
def test_worker_failure_matrix(
    monkeypatch: pytest.MonkeyPatch,
    operation_id: str,
    module: str,
    changes: dict[str, Any],
    expected: type[Exception],
) -> None:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    request = operation.request_type.model_validate(operation.examples[0].input)
    values: dict[str, Any] = {
        "returncode": 0,
        "stdout": b"{}",
        "stderr": b"",
        "stdout_exceeded": False,
        "stderr_exceeded": False,
        "timed_out": False,
    }
    values.update(changes)
    monkeypatch.setattr(
        importlib.import_module(module),
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(**values),
    )
    with pytest.raises(expected):
        operation.run(request)


@pytest.mark.parametrize(("operation_id", "module"), OWNERS)
def test_decode_cannot_finish_after_parent_deadline(
    monkeypatch: pytest.MonkeyPatch, operation_id: str, module: str
) -> None:
    import time

    from jacobian._execution import request_execution

    owner = importlib.import_module(module)
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    request = operation.request_type.model_validate(operation.examples[0].input)
    now = time.monotonic()
    clock = [now]

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        clock[0] += 1000
        return BoundedProcessResult(0, b"{}", b"", False, False, False)

    monkeypatch.setattr(owner, "run_bounded_process", complete)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    with request_execution(now), pytest.raises(OperationExecutionTimeoutError):
        operation.run(request)


@pytest.mark.parametrize(
    ("operation_id", "optimum", "witness_kind"),
    [
        ("graph.domination.minimum.compute", 2, "domination"),
        ("graph.matching.maximal.minimum.compute", 3, "matching"),
        ("graph.induced_forest.maximum.compute", 4, "forest"),
        ("graph.induced_tree.maximum.compute", 4, "tree"),
        ("graph.induced_bipartite.maximum.compute", 4, "bipartite"),
        ("graph.invariant.clique_number.compute", 2, "clique"),
    ],
)
def test_worker_search_limit_preserves_valid_partial_bounds(
    operation_id: str,
    optimum: int,
    witness_kind: str,
) -> None:
    from itertools import combinations

    import networkx as nx

    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    graph = nx.cycle_graph(list("abcdefg" if witness_kind == "matching" else "abcde"))
    request = operation.request_type.model_validate(
        {
            "graph": {"vertices": list(graph), "edges": list(graph.edges)},
            "resource_budget": {"wall_seconds": 5, "max_solver_calls": 1},
        }
    )
    result = operation.run(request)
    assert result.status == "UNKNOWN"
    assert result.optimum_value is None
    assert result.lower_bound <= optimum <= result.upper_bound
    if witness_kind == "matching":
        assert nx.is_maximal_matching(graph, result.witness_edges)
        assert result.incumbent_value == len(result.witness_edges)
    else:
        witness = result.witness_vertices
        assert set(witness) <= set(graph)
        assert result.incumbent_value == len(witness)
        if witness_kind == "domination":
            assert nx.is_dominating_set(graph, witness)
        elif witness_kind == "forest":
            assert not witness or nx.is_forest(graph.subgraph(witness))
        elif witness_kind == "tree":
            assert nx.is_tree(graph.subgraph(witness))
        elif witness_kind == "bipartite":
            assert nx.is_bipartite(graph.subgraph(witness))
        else:
            assert all(graph.has_edge(a, b) for a, b in combinations(witness, 2))


def test_hypergraph_preserves_unspent_dispatch_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import time

    from jacobian._execution import request_execution
    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3 as owner,
    )

    operation = Catalog.open().operation("hypergraph.independence_number.compute")
    assert operation is not None
    request = operation.request_type.model_validate(
        {**operation.examples[0].input, "resource_budget": {"wall_seconds": 10}}
    )
    expected = owner._solve_independence_number_kernel(
        request.hypergraph, request.resource_budget
    )
    projection = expected.model_dump(
        mode="json", exclude={"hypergraph", "resource_budget"}
    )
    recorded: dict[str, Any] = {}

    def complete(*args: object, **kwargs: Any) -> BoundedProcessResult:
        recorded.update(kwargs)
        return BoundedProcessResult(
            0, encode_worker_result_frame(projection), b"", False, False, False
        )

    monkeypatch.setattr(owner, "run_bounded_process", complete)
    monkeypatch.setattr(time, "monotonic", lambda: 106.0)
    with request_execution(100.0):
        assert operation.run(request) == expected
    assert 3.9 < recorded["timeout_seconds"] < 4.0
    assert json.loads(recorded["input_bytes"])["_deadline"] == pytest.approx(
        106.0 + recorded["timeout_seconds"]
    )


def test_hypergraph_error_decoding_preserves_parent_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import time

    from jacobian.math.combinatorics.finite_structures.hypergraphs import (
        _independence_z3 as owner,
    )

    now = [106.0]
    original_loads = json.loads

    def delayed_loads(*args: Any, **kwargs: Any) -> Any:
        response = original_loads(*args, **kwargs)
        now[0] = 111.0
        return response

    completed = BoundedProcessResult(
        0,
        b'{"kind":"execution_error","stage":"operation_execution","resource":"work"}',
        b"",
        False,
        False,
        False,
    )
    monkeypatch.setattr(owner, "run_bounded_process", lambda *args, **kwargs: completed)
    monkeypatch.setattr(json, "loads", delayed_loads)
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    with pytest.raises(OperationExecutionTimeoutError):
        owner._run_independence_worker({}, deadline=110.0)
