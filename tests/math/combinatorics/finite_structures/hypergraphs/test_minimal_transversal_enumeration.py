"""Complete bounded-cardinality minimal transversal enumeration."""

import time

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._transversal_enumeration import (
    MinimalTransversalEnumerationRequest,
    MinimalTransversalEnumerationResult,
    enumerate_minimal_transversals,
)


def test_crossing_edges_have_singleton_and_two_vertex_minima() -> None:
    source = FiniteHypergraph(
        vertices=("a", "b", "c"),
        edges=(("e0", ("a", "b")), ("e1", ("b", "c"))),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=2)
    )
    assert result.transversals == (("b",), ("a", "c"))
    assert [(row.cardinality, row.count) for row in result.cardinality_profile] == [
        (0, 0),
        (1, 1),
        (2, 1),
    ]


def test_edge_free_hypergraph_has_one_empty_minimal_transversal() -> None:
    source = FiniteHypergraph(vertices=("a", "b"), edges=())
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=0)
    )
    assert result.transversals == ((),)
    assert [(row.cardinality, row.count) for row in result.cardinality_profile] == [
        (0, 1)
    ]


def test_profile_retains_zero_ranks_above_source_cardinality() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("edge", ("a",)),))
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=3)
    )
    assert result.transversals == (("a",),)
    assert [(row.cardinality, row.count) for row in result.cardinality_profile] == [
        (0, 0),
        (1, 1),
        (2, 0),
        (3, 0),
    ]


def test_empty_edge_has_no_transversal() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("empty", ()),))
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=1)
    )
    assert result.transversals == ()
    assert [(row.cardinality, row.count) for row in result.cardinality_profile] == [
        (0, 0),
        (1, 0),
    ]


def test_result_validates_rows_structurally_without_replaying_hitting() -> None:
    source = FiniteHypergraph(
        vertices=("b", "a"),
        edges=(("e", ("a",)),),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=1)
    )
    payload = result.model_dump()
    payload["transversals"] = [["b"]]
    payload["cardinality_profile"] = [
        {"cardinality": 0, "count": 0},
        {"cardinality": 1, "count": 1},
    ]
    assert type(result).model_validate(payload).transversals == (("b",),)


def test_result_rejects_duplicate_or_noncanonical_rows() -> None:
    source = FiniteHypergraph(
        vertices=("a", "b", "c"),
        edges=(("e", ("a", "b")),),
    )
    base = {
        "hypergraph": source.model_dump(),
        "maximum_cardinality": 2,
        "cardinality_profile": [
            {"cardinality": 0, "count": 0},
            {"cardinality": 1, "count": 0},
            {"cardinality": 2, "count": 2},
        ],
    }
    with pytest.raises(ValidationError, match="canonical cardinality order"):
        MinimalTransversalEnumerationResult.model_validate(
            {**base, "transversals": [["a", "b"], ["a", "b"]]}
        )
    with pytest.raises(ValidationError, match="declared vertex order"):
        MinimalTransversalEnumerationResult.model_validate(
            {**base, "transversals": [["b", "a"], ["a", "c"]]}
        )


def test_antichain_row_bound_allows_candidate_slice_beyond_row_bound() -> None:
    source = FiniteHypergraph(
        vertices=tuple(f"v{index:02d}" for index in range(20)),
        edges=(("edge", tuple(f"v{index:02d}" for index in range(20))),),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=7)
    )
    assert result.transversals == tuple((vertex,) for vertex in source.vertices)


def test_direct_native_guard_rejects_model_constructed_request() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=())
    malformed = MinimalTransversalEnumerationRequest.model_construct(
        hypergraph=source, maximum_cardinality="1"
    )
    with pytest.raises(OperationDomainValidationError, match="malformed typed request"):
        enumerate_minimal_transversals(malformed)


def test_native_operation_honors_request_deadline_before_search() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("edge", ("a",)),))
    request = MinimalTransversalEnumerationRequest(
        hypergraph=source, maximum_cardinality=1
    )
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError, match="deadline expired"):
            enumerate_minimal_transversals(request)


def test_candidate_slice_is_admitted_before_materialization() -> None:
    # A dense candidate slice can exceed the result carrier even when the
    # source hypergraph itself is small.  This is rejected before combinations
    # are generated.
    source = FiniteHypergraph(
        vertices=tuple(f"v{index:02d}" for index in range(20)),
        edges=(("edge", tuple(f"v{index:02d}" for index in range(20))),),
    )
    with pytest.raises(OperationResourceAdmissionError, match="result rows"):
        enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=8
            )
        )


def test_candidate_edge_and_minimality_work_is_admitted_before_search() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(50))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(
            (f"edge{index:05d}", ("v00", "v01", "v02")) for index in range(8_000)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=3
            )
        )
