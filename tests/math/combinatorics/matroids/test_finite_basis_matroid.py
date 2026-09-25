"""Exact contracts for complete finite basis-matroid values."""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.matroids import FiniteBasisMatroid
from jacobian.math.combinatorics.matroids.values import (
    MAX_FINITE_BASIS_COUNT,
    MAX_FINITE_BASIS_EXCHANGE_CHECKS,
    MAX_FINITE_BASIS_GROUND_SIZE,
    MAX_FINITE_BASIS_LABEL_BYTES,
    MAX_FINITE_BASIS_MEMBERSHIPS,
    MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
)


def _satisfies_basis_exchange(bases: tuple[tuple[int, ...], ...]) -> bool:
    """Independent direct-set oracle for the ordinary basis axiom."""

    family = {frozenset(basis) for basis in bases}
    return all(
        any((left - {removed}) | {added} in family for added in right - left)
        for left in family
        for right in family
        for removed in left - right
    )


def test_all_small_equal_size_families_match_independent_basis_axiom() -> None:
    for ground_size in range(5):
        ground = tuple(f"e{index}" for index in range(ground_size))
        for rank in range(ground_size + 1):
            possible_bases = tuple(itertools.combinations(range(ground_size), rank))
            for family_mask in range(1, 1 << len(possible_bases)):
                bases = tuple(
                    possible_bases[index]
                    for index in range(len(possible_bases))
                    if family_mask >> index & 1
                )
                valid = _satisfies_basis_exchange(bases)
                value = FiniteBasisMatroid(ground=ground, bases=bases)
                assert value.bases == bases
                assert value.rank == rank
                assert value.ground == ground
                if valid:
                    value.require_basis_exchange()
                else:
                    with pytest.raises(Exception, match="basis exchange"):
                        value.require_basis_exchange()


@pytest.mark.parametrize(
    ("ground", "bases", "error"),
    [
        (("a", "b", "c", "d"), ((0, 1), (2, 3)), "basis_exchange"),
        (("a", "b", "c"), ((0,), (0, 1)), "basis_cardinality"),
        (("a", "b"), ((1, 0),), "basis_canonical"),
        (("a", "b"), ((0,), (0,)), "basis_family_canonical"),
        (("a", "a"), ((0,),), "ground_duplicate"),
        (("a",), ((True,),), "int_type"),
        (("a",), (), "at least 1 item"),
    ],
)
def test_rejects_noncanonical_or_nonmatroid_basis_families(
    ground: tuple[str, ...], bases: tuple[tuple[int, ...], ...], error: str
) -> None:
    if error == "basis_exchange":
        value = FiniteBasisMatroid(ground=ground, bases=bases)
        with pytest.raises(Exception, match="basis exchange"):
            value.require_basis_exchange()
    else:
        with pytest.raises(ValidationError, match=error):
            FiniteBasisMatroid(ground=ground, bases=bases)


def test_empty_ground_rank_zero_and_full_rank_preserve_the_ground_axis() -> None:
    empty = FiniteBasisMatroid(ground=(), bases=((),))
    loops = FiniteBasisMatroid(ground=("loop-a", "loop-b"), bases=((),))
    full_rank = FiniteBasisMatroid(ground=("a", "b", "c"), bases=((0, 1, 2),))

    assert (empty.rank, empty.ground, empty.bases) == (0, (), ((),))
    assert (loops.rank, loops.ground, loops.bases) == (
        0,
        ("loop-a", "loop-b"),
        ((),),
    )
    assert full_rank.rank == 3


def test_json_round_trip_is_structural_and_exchange_check_is_explicit() -> None:
    value = FiniteBasisMatroid(
        ground=("a", "b", "c", "d"),
        bases=tuple(itertools.combinations(range(4), 2)),
    )

    decoded = FiniteBasisMatroid.model_validate_json(value.model_dump_json())

    assert decoded == value
    assert decoded.rank == 2
    decoded.require_basis_exchange()


def test_schema_and_raw_envelope_publish_and_enforce_exact_limits() -> None:
    schema = FiniteBasisMatroid.model_json_schema()
    assert schema["properties"]["ground"]["maxItems"] == MAX_FINITE_BASIS_GROUND_SIZE
    assert schema["properties"]["ground"]["items"]["maxLength"] == (
        MAX_FINITE_BASIS_LABEL_BYTES
    )
    assert schema["properties"]["bases"]["minItems"] == 1
    assert schema["properties"]["bases"]["maxItems"] == MAX_FINITE_BASIS_COUNT
    assert schema["properties"]["bases"]["items"]["maxItems"] == (
        MAX_FINITE_BASIS_GROUND_SIZE
    )
    assert schema["admission_limits"] == {
        "max_ground_elements": MAX_FINITE_BASIS_GROUND_SIZE,
        "max_basis_rows": MAX_FINITE_BASIS_COUNT,
        "max_basis_memberships": MAX_FINITE_BASIS_MEMBERSHIPS,
        "max_ground_label_utf8_bytes_each": MAX_FINITE_BASIS_LABEL_BYTES,
        "max_ground_label_utf8_bytes_total": MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
        "max_basis_exchange_candidate_checks": MAX_FINITE_BASIS_EXCHANGE_CHECKS,
    }

    max_ground = tuple(f"e{index}" for index in range(MAX_FINITE_BASIS_GROUND_SIZE))
    rank_zero = FiniteBasisMatroid(ground=max_ground, bases=((),))
    assert rank_zero.ground_size == MAX_FINITE_BASIS_GROUND_SIZE

    with pytest.raises(ValidationError, match="ground_size_bound"):
        FiniteBasisMatroid.model_validate(
            {"ground": [*max_ground, "too-many"], "bases": [[]]}
        )

    over_membership_family = [list(range(MAX_FINITE_BASIS_GROUND_SIZE))] * 1_025
    with pytest.raises(ValidationError, match="membership_bound"):
        FiniteBasisMatroid.model_validate(
            {"ground": max_ground, "bases": over_membership_family}
        )

    work_overflow = tuple(itertools.islice(itertools.combinations(range(64), 32), 45))
    bounded_claim = FiniteBasisMatroid(ground=max_ground, bases=work_overflow)
    with pytest.raises(Exception, match="work exceeds the admitted bound"):
        bounded_claim.require_basis_exchange()


def test_raw_json_oversized_nested_family_fails_before_model_construction() -> None:
    ground = [f"e{index}" for index in range(64)]
    bases = [list(range(64))] * 1_025
    raw_json = json.dumps({"ground": ground, "bases": bases})

    with pytest.raises(ValidationError, match="membership_bound"):
        FiniteBasisMatroid.model_validate_json(raw_json)


def test_utf8_label_limits_apply_per_label_and_in_aggregate() -> None:
    per_label_limit = "🦊" * (MAX_FINITE_BASIS_LABEL_BYTES // 4)
    accepted_single = FiniteBasisMatroid(ground=(per_label_limit,), bases=((),))
    assert accepted_single.ground == (per_label_limit,)

    with pytest.raises(ValidationError, match="ground_label_bound"):
        FiniteBasisMatroid(ground=(per_label_limit + "🦊",), bases=((),))

    exact_aggregate = tuple(
        f"{index}:" + "x" * (MAX_FINITE_BASIS_LABEL_BYTES - len(f"{index}:"))
        for index in range(
            MAX_FINITE_BASIS_TOTAL_LABEL_BYTES // MAX_FINITE_BASIS_LABEL_BYTES
        )
    )
    assert sum(len(label.encode("utf-8")) for label in exact_aggregate) == (
        MAX_FINITE_BASIS_TOTAL_LABEL_BYTES
    )
    FiniteBasisMatroid(ground=exact_aggregate, bases=((),))

    over_aggregate = (*exact_aggregate, "new-label")
    with pytest.raises(ValidationError, match="ground_label_total_bound"):
        FiniteBasisMatroid.model_validate_json(
            json.dumps({"ground": over_aggregate, "bases": [[]]})
        )
