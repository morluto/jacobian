"""Regressions for the delta-matroid native graph admission.

Each repaired case is paired with a negative control showing the enclosing
bound or domain rejection is unchanged.
"""

from __future__ import annotations

import time
from itertools import combinations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta.extra import MAX_BINARY_GROUND
from jacobian.math.graphs.delta_matroids._models import (
    LoopedGraphDeltaMatroidResult,
    admit_looped_graph,
)
from jacobian.math.graphs.delta_matroids._tools import TOOLS
from jacobian.math.graphs.delta_matroids.operations import (
    looped_adjacency_delta_matroid,
)
from jacobian.math.graphs.values import LoopedSimpleGraph


def _canonical(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...] = (),
    loops: tuple[str, ...] = (),
) -> LoopedSimpleGraph:
    return LoopedSimpleGraph.model_validate(
        {
            "vertices": list(vertices),
            "edges": [list(edge) for edge in edges],
            "loops": list(loops),
        }
    )


def _forged(
    vertices: object, edges: object = (), loops: object = ()
) -> LoopedSimpleGraph:
    return LoopedSimpleGraph.model_construct(
        vertices=vertices, edges=edges, loops=loops
    )


def _code(error: OperationDomainValidationError) -> str:
    return str(error.errors()[0]["type"])


# --- the vertex count is refused before the carrier is copied --------------


def test_oversized_forged_carrier_is_refused_as_a_resource() -> None:
    """An over-large request is a resource rejection, not a malformed one."""
    forged = _forged(tuple(f"v{index}" for index in range(200_000)))
    start = time.perf_counter()
    with pytest.raises(OperationResourceAdmissionError) as error:
        admit_looped_graph(forged)
    elapsed = time.perf_counter() - start
    assert _code(error.value) == "delta_matroid.binary_work"
    # The preflight is an O(1) length check, so it does not serialize first.
    assert elapsed < 0.5


def test_just_over_the_vertex_limit_is_refused_as_a_resource() -> None:
    forged = _forged(tuple(f"v{index}" for index in range(MAX_BINARY_GROUND + 1)))
    with pytest.raises(OperationResourceAdmissionError) as error:
        admit_looped_graph(forged)
    assert _code(error.value) == "delta_matroid.binary_work"


# --- the raw container is preflighted before serialization ------------------


def test_non_tuple_vertex_container_is_refused() -> None:
    forged = _forged(["a", "b"])
    with pytest.raises(OperationDomainValidationError) as error:
        admit_looped_graph(forged)
    assert _code(error.value) == "graph.looped_graph_invalid"


@pytest.mark.parametrize(
    "edges",
    [
        7,
        (("a",),),
        ("ab",),
    ],
    ids=["not-iterable", "row-not-a-pair", "row-not-a-container"],
)
def test_malformed_edge_containers_keep_the_domain_error_classification(
    edges: object,
) -> None:
    """A malformed carrier never leaks a serialization helper exception.

    A list of well-formed pairs is deliberately absent: the canonical rebuild
    normalises it, which is the documented trust-boundary behaviour.
    """
    forged = _forged(("a", "b"), edges=edges, loops=("a",))
    with pytest.raises(OperationDomainValidationError) as error:
        admit_looped_graph(forged)
    assert _code(error.value) == "graph.looped_graph_invalid"


# --- the published example states its input precondition --------------------


def test_example_states_its_input_precondition() -> None:
    tool = next(item for item in TOOLS if "delta_matroid" in item.operation_id)
    description = tool.examples[0].description
    assert "canonical" in description
    assert "label order" in description
    assert str(MAX_BINARY_GROUND) in description


# --- negative controls -----------------------------------------------------


def test_valid_looped_graph_is_admitted() -> None:
    graph = _canonical(("a", "b"), (("a", "b"),), ("a",))
    admitted = admit_looped_graph(graph)
    assert admitted.vertices == ("a", "b")


def test_exactly_at_the_vertex_limit_is_admitted() -> None:
    graph = _canonical(tuple(f"v{index}" for index in range(MAX_BINARY_GROUND)))
    admitted = admit_looped_graph(graph)
    assert len(admitted.vertices) == MAX_BINARY_GROUND


def test_a_non_carrier_is_still_refused() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        admit_looped_graph({"vertices": ["a"]})  # type: ignore[arg-type]
    assert _code(error.value) == "graph.looped_graph_invalid"


def test_a_canonical_graph_still_revalidates() -> None:
    """A forged but small carrier is still rebuilt, not trusted in place."""
    forged = _forged(("a", "b", "c"), edges=(("a", "b"),), loops=())
    admitted = admit_looped_graph(forged)
    assert admitted.vertices == ("a", "b", "c")
    assert type(admitted.vertices) is tuple


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("edges", (("a", "b"),) * 100_000),
        ("loops", ("a",) * 100_000),
        ("edges", [("a", "b")] * 100_000),
        ("loops", ["a"] * 100_000),
        ("edges", (("a",) * 100_000,)),
        ("edges", (("a", ["b"] * 100_000),)),
        ("vertices", ("a", "b" * 100_000)),
        ("edges", (("a", "b" * 100_000),)),
        ("loops", ("a" * 100_000,)),
    ],
)
def test_all_raw_graph_storage_is_bounded_before_serialization(
    monkeypatch: pytest.MonkeyPatch, field: str, payload: object
) -> None:
    graph = _forged(("a", "b"))
    graph = graph.model_copy(update={field: payload})

    def unexpected_dump(*args: object, **kwargs: object) -> None:
        pytest.fail("unadmitted graph storage reached serialization")

    monkeypatch.setattr(LoopedSimpleGraph, "model_dump", unexpected_dump)
    with pytest.raises(OperationDomainValidationError) as error:
        looped_adjacency_delta_matroid(graph)
    assert _code(error.value) == "graph.looped_graph_invalid"


class _DeceptiveTuple(tuple[object, ...]):
    def __len__(self) -> int:
        pytest.fail("caller-controlled length hook was invoked")


@pytest.mark.parametrize("field", ["vertices", "edges", "loops", "edge_row"])
def test_raw_container_subclasses_are_refused_without_running_hooks(field: str) -> None:
    forged = _forged(("a", "b"))
    payload = _DeceptiveTuple(("a", "b"))
    forged = forged.model_copy(
        update={"edges": (payload,)} if field == "edge_row" else {field: payload}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        looped_adjacency_delta_matroid(forged)
    assert _code(error.value) == "graph.looped_graph_invalid"


@pytest.mark.parametrize("order", [0, MAX_BINARY_GROUND])
def test_dense_graph_at_derived_storage_limits_roundtrips_and_composes(
    order: int,
) -> None:
    vertices = tuple(f"v{index}" for index in range(order))
    graph = _canonical(vertices, tuple(combinations(vertices, 2)), vertices)
    result = looped_adjacency_delta_matroid(graph)
    # Every nonempty principal matrix is all ones: only singletons have full rank.
    assert result.matrix.entries == ((1,) * order,) * order
    assert result.delta_matroid.feasible == ((), *((i,) for i in range(order)))
    decoded = LoopedGraphDeltaMatroidResult.model_validate_json(
        result.model_dump_json()
    )
    assert looped_adjacency_delta_matroid(decoded.graph) == result


def test_small_list_edge_storage_still_normalizes_to_canonical_tuples() -> None:
    graph = _forged(("a", "b"), edges=[["a", "b"]], loops=["a"])
    assert admit_looped_graph(graph) == _canonical(("a", "b"), (("a", "b"),), ("a",))
