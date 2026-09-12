"""Complete bounded-cardinality minimal transversal enumeration."""

import time
from math import comb
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import (
    _transversal_enumeration as enumeration,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._transversal_enumeration import (
    MAX_TRANSVERSAL_ENUMERATION_WORK,
    MinimalTransversalEnumerationRequest,
    MinimalTransversalEnumerationResult,
    _candidate_has_redundant_vertex,
    _candidate_hits_all_edges,
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


def test_accepted_near_envelope_execution_charges_each_search_primitive() -> None:
    vertex_count = 20
    maximum_cardinality = 6
    vertices = tuple(f"v{index:02d}" for index in range(vertex_count))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple((f"edge{index:03d}", vertices) for index in range(126)),
    )
    candidate_count = sum(
        comb(vertex_count, size) for size in range(1, maximum_cardinality + 1)
    )
    weighted_candidate_count = sum(
        size * comb(vertex_count, size) for size in range(1, maximum_cardinality + 1)
    )
    charged = {
        "candidate_edge": candidate_count * len(source.edges),
        "minimality": weighted_candidate_count * len(source.edges),
    }
    assert 100 * sum(charged.values()) >= 95 * MAX_TRANSVERSAL_ENUMERATION_WORK
    assert sum(charged.values()) <= MAX_TRANSVERSAL_ENUMERATION_WORK

    executed = dict.fromkeys(charged, 0)

    def count_candidate_edge_checks(*args: Any, **kwargs: Any) -> tuple[bool, int]:
        result = _candidate_hits_all_edges(*args, **kwargs)
        executed["candidate_edge"] += result[1]
        return result

    def count_minimality_checks(*args: Any, **kwargs: Any) -> tuple[bool, int]:
        result = _candidate_has_redundant_vertex(*args, **kwargs)
        executed["minimality"] += result[1]
        return result

    request = MinimalTransversalEnumerationRequest(
        hypergraph=source, maximum_cardinality=maximum_cardinality
    )
    with (
        patch.object(
            enumeration,
            "_candidate_hits_all_edges",
            side_effect=count_candidate_edge_checks,
        ),
        patch.object(
            enumeration,
            "_candidate_has_redundant_vertex",
            side_effect=count_minimality_checks,
        ),
    ):
        result = enumerate_minimal_transversals(request)

    assert result.transversals == tuple((vertex,) for vertex in vertices)
    assert set(executed) == set(charged)
    assert_charged_work_parity(charged=charged, executed=executed)
