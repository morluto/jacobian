"""Regressions for the delta-matroid native graph admission.

Each repaired case is paired with a negative control showing the enclosing
bound or domain rejection is unchanged.
"""

from __future__ import annotations

import time

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta.extra import MAX_BINARY_GROUND
from jacobian.math.graphs.delta_matroids._models import admit_looped_graph
from jacobian.math.graphs.delta_matroids._tools import TOOLS
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
