"""Defining contracts for finite-Abelian zero-sum atom hypergraphs."""

from __future__ import annotations

from itertools import combinations
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import MathTool
from jacobian.math.combinatorics.additive.zero_sum_atoms import (
    construct_zero_sum_atom_hypergraph,
    verify_zero_sum_atom,
    verify_zero_sum_atom_hypergraph,
)
from jacobian.math.combinatorics.additive.zero_sum_atoms._models import (
    MAX_ATOM_EDGES,
    MAX_ATOM_GROUP_ORDER,
    MAX_ATOM_INCIDENCES,
    MAX_ATOM_SOURCE_ELEMENTS,
    MAX_ATOM_SUBSET_CHECKS,
    ZeroSumAtomHypergraphRequest,
    ZeroSumAtomHypergraphResult,
    ZeroSumAtomSource,
)
from jacobian.math.combinatorics.additive.zero_sum_atoms._tools import TOOLS
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    MaximumEdgeMatchingRequest,
    MinimumTransversalRequest,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    maximum_edge_matching,
    minimum_transversal,
)
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup


def _operation() -> MathTool[ZeroSumAtomHypergraphRequest, ZeroSumAtomHypergraphResult]:
    return cast(
        MathTool[ZeroSumAtomHypergraphRequest, ZeroSumAtomHypergraphResult],
        next(
            operation
            for operation in TOOLS
            if operation.operation_id == "additive.zero_sum.atom_hypergraph.construct"
        ),
    )


def _source(
    elements: tuple[tuple[int, ...], ...],
    moduli: tuple[int, ...],
) -> ZeroSumAtomSource:
    return ZeroSumAtomSource(
        group=FiniteAbelianProductGroup(moduli=moduli), elements=elements
    )


def _run(
    elements: tuple[tuple[int, ...], ...],
    moduli: tuple[int, ...],
) -> ZeroSumAtomHypergraphResult:
    return construct_zero_sum_atom_hypergraph(_source(elements, moduli))


def _zero_sum(
    element_indices: tuple[int, ...],
    source: ZeroSumAtomSource,
) -> bool:
    running = tuple(0 for _ in source.group.moduli)
    for index in element_indices:
        running = tuple(
            (left + right) % modulus
            for left, right, modulus in zip(
                running,
                source.elements[index],
                source.group.moduli,
                strict=True,
            )
        )
    return running == tuple(0 for _ in source.group.moduli)


def _edge_indices(result: ZeroSumAtomHypergraphResult) -> tuple[frozenset[int], ...]:
    return tuple(frozenset(map(int, members)) for _, members in result.hypergraph.edges)


def test_empty_source_has_empty_atom_hypergraph() -> None:
    result = _run((), (7,))

    assert result.hypergraph.vertices == ()
    assert result.hypergraph.edges == ()
    assert result.atom_count == 0
    assert result.total_incidences == 0


def test_serialized_forged_atom_family_is_rejected_by_verifier() -> None:
    result = _run(((1,), (6,)), (7,))
    payload = result.model_dump(mode="json")
    payload["hypergraph"]["edges"] = []
    payload["atom_count"] = 0
    payload["total_incidences"] = 0
    decoded = ZeroSumAtomHypergraphResult.model_validate_json(
        encode_strict_json(payload)
    )
    assert not verify_zero_sum_atom_hypergraph(decoded)


def test_atom_verifier_checks_zero_sum_and_inclusion_minimality() -> None:
    source = _source(((1,), (3,), (5,)), (6,))
    assert verify_zero_sum_atom(source, (0, 2))
    assert not verify_zero_sum_atom(source, (0, 1, 2))
    assert not verify_zero_sum_atom(source, (0, 1))


def test_zero_only_source_has_identity_singleton_atom() -> None:
    result = _run(((0,),), (7,))

    assert result.hypergraph.vertices == ("0",)
    assert result.hypergraph.edges == (("0", ("0",)),)
    assert result.atom_count == 1
    assert result.total_incidences == 1


def test_zero_sum_free_source_has_no_edges() -> None:
    result = _run(((1,), (2,)), (7,))

    assert result.hypergraph.vertices == ("0", "1")
    assert result.hypergraph.edges == ()


def test_one_inverse_pair_is_an_atom() -> None:
    result = _run(((1,), (6,)), (7,))

    assert result.hypergraph.vertices == ("0", "1")
    assert result.hypergraph.edges == (("0,1", ("0", "1")),)
    assert result.atom_count == 1
    assert result.total_incidences == 2


def test_nonminimal_zero_sum_contains_smaller_atom() -> None:
    # In Z/6Z, {1,5} is a zero-sum atom and {1,3,5} is zero-sum but contains it.
    result = _run(((1,), (3,), (5,)), (6,))

    assert _edge_indices(result) == (frozenset({0, 2}),)


def test_full_z7_atom_family_is_complete() -> None:
    result = _run(((1,), (2,), (3,), (4,), (5,), (6,)), (7,))

    expected = {
        frozenset({0, 5}),
        frozenset({1, 4}),
        frozenset({2, 3}),
        frozenset({0, 1, 3}),
        frozenset({2, 4, 5}),
    }
    assert set(_edge_indices(result)) == expected
    assert result.atom_count == 5
    assert result.total_incidences == 12


def test_product_group_coordinates_and_parent_moduli_survive() -> None:
    source = _source(((0, 1), (1, 0), (1, 2), (2, 1)), (3, 4))
    result = construct_zero_sum_atom_hypergraph(source)

    assert result.source == source
    assert result.source.group.moduli == (3, 4)
    assert result.hypergraph.vertices == ("0", "1", "2", "3")
    for _, members in result.hypergraph.edges:
        indices = tuple(map(int, members))
        assert _zero_sum(indices, source)


def test_source_reduces_rows_and_rejects_duplicates_after_reduction() -> None:
    source = _source(
        (
            (7,),
            (2,),
        ),
        (7,),
    )
    assert source.elements == ((0,), (2,))

    with pytest.raises(ValidationError, match="distinct and sorted"):
        ZeroSumAtomSource.model_validate_json(
            encode_strict_json(
                {
                    "group": {"moduli": ["7"]},
                    "elements": [[7], [0]],
                }
            )
        )


@pytest.mark.parametrize(
    "source",
    (
        {"group": {"moduli": [[7]]}, "elements": []},
        {"group": {"moduli": ["7"]}, "elements": [[[1]]]},
    ),
)
def test_source_rejects_nested_raw_scalars(source: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"integer scalars|decimal strings"):
        ZeroSumAtomSource.model_validate_json(encode_strict_json(source))


@pytest.mark.parametrize(
    "source",
    (
        {"group": {"moduli": ["7"]}, "elements": [], "extra": [[0] * 100]},
        {"group": {"moduli": [7], "extra": [[0] * 100]}, "elements": []},
    ),
)
def test_source_rejects_unknown_fields_before_canonicalization(
    source: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="unknown fields"):
        ZeroSumAtomSource.model_validate_json(encode_strict_json(source))


def test_exhaustive_small_sources_match_independent_oracle() -> None:
    moduli = (7,)
    values = (0, 1, 2, 3, 4, 5, 6)
    for size in range(5):
        for selected in combinations(values, size):
            source = _source(tuple((value,) for value in selected), moduli)
            result = construct_zero_sum_atom_hypergraph(source)
            expected: set[frozenset[int]] = set()
            indices = tuple(range(len(source.elements)))
            for edge_size in range(1, len(indices) + 1):
                for candidate in combinations(indices, edge_size):
                    if not _zero_sum(candidate, source):
                        continue
                    if any(atom < set(candidate) for atom in expected):
                        continue
                    expected.add(frozenset(candidate))
            assert set(_edge_indices(result)) == expected


def test_every_edge_replays_zero_sum_and_minimality() -> None:
    source = _source(((0,), (1,), (2,), (3,), (4,), (5,), (6,)), (7,))
    result = construct_zero_sum_atom_hypergraph(source)

    for _, members in result.hypergraph.edges:
        indices = tuple(map(int, members))
        assert _zero_sum(indices, source)
        for proper_size in range(1, len(indices)):
            for proper in combinations(indices, proper_size):
                assert not _zero_sum(proper, source)


def test_result_round_trip_and_catalog_example_execute() -> None:
    operation = _operation()
    request = ZeroSumAtomHypergraphRequest.model_validate_json(
        encode_strict_json(operation.examples[0].input)
    )
    result = operation.run(request)

    decoded = ZeroSumAtomHypergraphResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert decoded == result
    assert decoded.hypergraph == FiniteHypergraph.model_validate_json(
        encode_strict_json(decoded.hypergraph.model_dump(mode="json"))
    )


def test_projection_composes_with_transversal_and_matching_consumers() -> None:
    result = _run(((1,), (2,), (3,), (4,), (5,), (6,)), (7,))

    transversal = minimum_transversal(
        MinimumTransversalRequest.model_validate_json(
            encode_strict_json(
                {"hypergraph": result.hypergraph.model_dump(mode="json")}
            )
        ).hypergraph
    )
    matching = maximum_edge_matching(
        MaximumEdgeMatchingRequest.model_validate_json(
            encode_strict_json(
                {"hypergraph": result.hypergraph.model_dump(mode="json")}
            )
        ).hypergraph
    )

    assert transversal.cardinality == 3
    assert matching.count == 3


def test_admission_accepts_cheap_large_group_sources_and_result_boundaries() -> None:
    assert _run((), (4097,)).atom_count == 0

    with pytest.raises(ValidationError, match="at most 24 items"):
        ZeroSumAtomSource.model_validate_json(
            encode_strict_json(
                {
                    "group": {"moduli": ["7"]},
                    "elements": [[value] for value in range(25)],
                }
            )
        )

    # The 24-element source ceiling is exactly the largest complete subset
    # tree inside the 20,000,000-check bound: 2**24 = 16,777,216.
    assert (1 << MAX_ATOM_SOURCE_ELEMENTS) <= MAX_ATOM_SUBSET_CHECKS
    assert (1 << (MAX_ATOM_SOURCE_ELEMENTS + 1)) > MAX_ATOM_SUBSET_CHECKS

    assert MAX_ATOM_SOURCE_ELEMENTS == 24
    assert MAX_ATOM_GROUP_ORDER == 4_096
    assert MAX_ATOM_EDGES == 12_000
    assert MAX_ATOM_INCIDENCES == 36_000


def test_result_rejects_forged_projection_and_counts() -> None:
    result = _run(((1,), (6,)), (7,))
    payload = result.model_dump(mode="json")

    with pytest.raises(ValidationError):
        ZeroSumAtomHypergraphResult.model_validate_json(
            encode_strict_json({**payload, "atom_count": result.atom_count + 1})
        )
    with pytest.raises(ValidationError):
        ZeroSumAtomHypergraphResult.model_validate_json(
            encode_strict_json(
                {**payload, "total_incidences": result.total_incidences + 1}
            )
        )
    with pytest.raises(ValidationError):
        ZeroSumAtomHypergraphResult.model_validate_json(
            encode_strict_json(
                {
                    **payload,
                    "hypergraph": {
                        **payload["hypergraph"],
                        "vertices": ["1", "0"],
                    },
                }
            )
        )
