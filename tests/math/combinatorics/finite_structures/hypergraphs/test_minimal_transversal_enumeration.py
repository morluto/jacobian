"""Complete bounded-cardinality minimal transversal enumeration."""

import time
from itertools import combinations
from math import comb
from typing import Any
from unittest.mock import patch

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


def test_single_edge_slice_uses_closed_form_above_global_row_bound() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(vertices=vertices, edges=(("edge", vertices),))
    request = MinimalTransversalEnumerationRequest(
        hypergraph=source, maximum_cardinality=8
    )
    with patch.object(
        enumeration,
        "combinations",
        side_effect=AssertionError("single-edge slice must use its closed form"),
    ):
        result = enumerate_minimal_transversals(request)

    assert result.transversals == tuple((vertex,) for vertex in vertices)
    assert [(row.cardinality, row.count) for row in result.cardinality_profile] == [
        (0, 0),
        (1, 20),
        *[(cardinality, 0) for cardinality in range(2, 9)],
    ]


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


def test_native_operation_checks_deadline_after_result_validation() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("edge", ("a",)),))
    request = MinimalTransversalEnumerationRequest(
        hypergraph=source, maximum_cardinality=1
    )
    original_result = enumeration.MinimalTransversalEnumerationResult
    clock = [0.0]

    def expire_after_validation(*args: Any, **kwargs: Any) -> Any:
        result = original_result(*args, **kwargs)
        clock[0] = 1.0
        return result

    with (
        patch.object(time, "monotonic", side_effect=lambda: clock[0]),
        request_execution(started_at=0.0, outer_deadline=0.5),
        patch.object(
            enumeration,
            "MinimalTransversalEnumerationResult",
            side_effect=expire_after_validation,
        ),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="after minimal transversal result validation",
        ),
    ):
        enumerate_minimal_transversals(request)


def test_candidate_slice_is_admitted_before_materialization() -> None:
    # Distinct edges keep the Sperner row bound; duplicate member-sets do not.
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=(
            ("edge0", vertices[:19]),
            ("edge1", vertices[1:]),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="result rows"):
        enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=8
            )
        )


def test_candidate_edge_and_minimality_work_is_admitted_before_search() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    triples = tuple(combinations(vertices, 3))[:200]
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(
            (f"edge{index:05d}", triple) for index, triple in enumerate(triples)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=6
            )
        )


def test_minimality_work_bound_is_enforced_after_candidate_work_fits() -> None:
    vertex_count = 20
    maximum_cardinality = 6
    vertices = tuple(f"v{index:02d}" for index in range(vertex_count))
    triples = tuple(combinations(vertices, 3))[:160]
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(
            (f"edge{index:03d}", triple) for index, triple in enumerate(triples)
        ),
    )
    edge_count = len(triples)
    candidate_count = sum(
        comb(vertex_count, size) for size in range(1, maximum_cardinality + 1)
    )
    weighted_candidate_count = sum(
        size * comb(vertex_count, size) for size in range(1, maximum_cardinality + 1)
    )
    candidate_edge_work = candidate_count * edge_count
    total_work = (candidate_count + weighted_candidate_count) * edge_count
    assert candidate_edge_work < MAX_TRANSVERSAL_ENUMERATION_WORK
    assert total_work > MAX_TRANSVERSAL_ENUMERATION_WORK

    with pytest.raises(
        OperationResourceAdmissionError,
        match=f"{total_work} checks; maximum is {MAX_TRANSVERSAL_ENUMERATION_WORK}",
    ):
        enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=maximum_cardinality
            )
        )


def test_accepted_near_envelope_execution_charges_each_search_primitive() -> None:
    vertex_count = 20
    maximum_cardinality = 6
    vertices = tuple(f"v{index:02d}" for index in range(vertex_count))
    triples = tuple(combinations(vertices, 3))[:126]
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(
            (f"edge{index:03d}", triple) for index, triple in enumerate(triples)
        ),
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

    assert result.transversals
    assert executed["candidate_edge"] > 0
    assert executed["minimality"] > 0
    assert executed["candidate_edge"] <= charged["candidate_edge"]
    assert executed["minimality"] <= charged["minimality"]


def test_duplicate_full_edges_dedup_to_the_single_edge_shortcut() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple((f"e{index}", vertices) for index in range(500)),
    )
    with patch.object(
        enumeration,
        "combinations",
        side_effect=AssertionError(
            "duplicate edges must not charge combinatorial search"
        ),
    ):
        result = enumerate_minimal_transversals(
            MinimalTransversalEnumerationRequest(
                hypergraph=source, maximum_cardinality=6
            )
        )
    assert result.transversals == tuple((vertex,) for vertex in vertices)


def test_singleton_presolve_empties_an_overconstrained_rank_slice() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=tuple((f"s{index}", (vertices[index],)) for index in range(9)),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=8)
    )
    assert result.transversals == ()


def test_owner_deadline_binds_inside_a_later_outer_deadline() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("edge", ("a",)),))
    request = MinimalTransversalEnumerationRequest(
        hypergraph=source, maximum_cardinality=1
    )
    bound: list[float] = []

    def capture(deadline: float) -> None:
        bound.append(deadline)
        bind_request_deadline(deadline)

    with (
        patch.object(time, "monotonic", return_value=100.0),
        request_execution(started_at=100.0, outer_deadline=100_000.0),
        patch(
            "jacobian._execution.bind_request_deadline",
            side_effect=capture,
        ),
    ):
        result = enumerate_minimal_transversals(request)
    assert result.transversals == (("a",),)
    assert bound
    assert bound[0] == pytest.approx(3_700.0)


def test_isolated_vertices_are_excluded_from_the_admission_universe() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=(("e0", (vertices[0], vertices[1])), ("e1", (vertices[1], vertices[2]))),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=8)
    )
    assert result.transversals == ((vertices[1],), (vertices[0], vertices[2]))


def test_dominated_edges_reduce_to_the_single_edge_shortcut() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(20))
    source = FiniteHypergraph(
        vertices=vertices,
        edges=(("small", vertices[:19]), ("large", vertices)),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=8)
    )
    assert result.transversals == tuple((vertex,) for vertex in vertices[:19])


def test_forced_vertices_are_included_in_minimality_search() -> None:
    source = FiniteHypergraph(
        vertices=("a", "b", "c", "d", "e"),
        edges=(
            ("ab", ("a", "b")),
            ("bc", ("b", "c")),
            ("d", ("d",)),
            ("e", ("e",)),
        ),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=4)
    )
    assert result.transversals == (("b", "d", "e"), ("a", "c", "d", "e"))
