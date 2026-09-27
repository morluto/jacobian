"""Tests for the closed exact common-basis operation."""

from __future__ import annotations

import pytest

from jacobian.math.combinatorics.matroids import (
    LinearMatroid,
    matroid_common_basis,
    verify_common_basis_result,
)
from jacobian.math.combinatorics.matroids._models import (
    MatroidCommonBasisRequest,
    MatroidCommonBasisResult,
)
from jacobian.math.combinatorics.matroids._tools import TOOLS
from jacobian.math.combinatorics.matroids.intersection import (
    replay_common_basis_result,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(
    rows: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def test_identical_triangle_has_a_common_basis() -> None:
    labels = ("a", "b", "c")
    triangle = _matroid(((1, 0, 1), (0, 1, 1)), labels)
    result = matroid_common_basis(triangle, triangle)

    assert result.status == "COMMON_BASIS"
    assert result.reason == "COMMON_BASIS"
    assert result.rank_first == result.rank_second == result.cardinality == 2
    assert result.common_basis == result.common_independent
    assert result.rank_first_common == result.rank_second_common == 2
    assert result.witness.equality == 2
    assert verify_common_basis_result(result)


def test_equal_rank_matroids_without_common_basis_have_exact_deficit() -> None:
    labels = ("a", "b")
    first = _matroid(((1, 0),), labels)
    second = _matroid(((0, 1),), labels)
    result = matroid_common_basis(first, second)

    assert result.status == "NO_COMMON_BASIS"
    assert result.reason == "MAXIMUM_COMMON_INDEPENDENT_SET_TOO_SMALL"
    assert result.rank_first == result.rank_second == 1
    assert result.cardinality == 0
    assert result.common_basis is None
    assert result.witness.equality == result.cardinality
    assert verify_common_basis_result(result)


def test_unequal_source_ranks_explain_no_common_basis() -> None:
    labels = ("a", "b", "c")
    rank_one = _matroid(((1, 0, 0),), labels)
    rank_two = _matroid(((1, 0, 1), (0, 1, 0)), labels)
    result = matroid_common_basis(rank_one, rank_two)

    assert result.status == "NO_COMMON_BASIS"
    assert result.reason == "SOURCE_RANK_MISMATCH"
    assert result.rank_first == 1
    assert result.rank_second == 2
    assert result.common_basis is None
    assert result.witness.equality == result.cardinality
    assert verify_common_basis_result(result)


def test_empty_ground_has_the_empty_common_basis() -> None:
    empty = _matroid(((),), ())
    result = matroid_common_basis(empty, empty)

    assert result.status == "COMMON_BASIS"
    assert result.common_basis == result.common_independent == ()
    assert result.rank_first == result.rank_second == result.cardinality == 0


def test_serialized_claim_decodes_without_replaying_minmax_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    labels = ("a", "b", "c")
    triangle = _matroid(((1, 0, 1), (0, 1, 1)), labels)
    result = matroid_common_basis(triangle, triangle)
    calls = 0
    original_rank = intersection.pf_rank

    def count_rank(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(intersection, "pf_rank", count_rank)
    assert (
        MatroidCommonBasisResult.model_validate_json(result.model_dump_json()) == result
    )
    assert calls == 0

    forged = result.model_dump(mode="python")
    forged["witness"] = {
        "subset": (0,),
        "rank_first": 1,
        "rank_second_complement": 1,
        "equality": result.cardinality,
    }
    decoded = MatroidCommonBasisResult.model_validate(forged)
    with pytest.raises(ValueError, match="rank claims"):
        replay_common_basis_result(decoded)


def test_mutated_source_rank_and_outcome_do_not_verify() -> None:
    labels = ("a", "b", "c")
    triangle = _matroid(((1, 0, 1), (0, 1, 1)), labels)
    result = matroid_common_basis(triangle, triangle)
    forged = MatroidCommonBasisResult.model_construct(
        **{
            **result.model_dump(mode="python"),
            "rank_first": 1,
        }
    )
    assert not verify_common_basis_result(forged)


def test_public_examples_return_closed_exact_outcomes() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "matroid.intersection.common_basis.compute"
    )
    assert len(tool.examples) == 2
    outcomes = []
    for example in tool.examples:
        request = MatroidCommonBasisRequest.model_validate(example.input)
        outcomes.append(tool.run(request).status)
    assert outcomes == ["COMMON_BASIS", "NO_COMMON_BASIS"]
