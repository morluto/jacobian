from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.gowers_cube_profile.operations import (
    compute_gowers_cube_profile,
    verify_gowers_cube_profile,
)


def _direct_cube_count(modulus: int, subset: tuple[int, ...], order: int) -> int:
    subset_values = set(subset)
    return sum(
        all(
            (x + sum(directions[bit] for bit in range(order) if mask & (1 << bit)))
            % modulus
            in subset_values
            for mask in range(1 << order)
        )
        for x in range(modulus)
        for directions in product(range(modulus), repeat=order)
    )


def test_full_set() -> None:
    """Full set of Z/5Z should have maximum cube count."""
    result = compute_gowers_cube_profile(5, (0, 1, 2, 3, 4), 1)
    # Order 1: cubes are (x, x+e) with both in A. Full set -> all 5*5 = 25
    assert result.cube_count == 25
    assert result.normalized_count.as_fraction() == Fraction(1)


def test_empty_set() -> None:
    result = compute_gowers_cube_profile(5, (), 1)
    assert result.cube_count == 0


def test_empty_set_short_circuits_large_generic_work_bound() -> None:
    result = compute_gowers_cube_profile(1001, (), 1)
    assert result.cube_count == 0


def test_single_element() -> None:
    result = compute_gowers_cube_profile(7, (3,), 1)
    # Only cube with both vertices at 3: base=3, e=0 -> (3,3)
    assert result.cube_count == 1
    assert result.normalized_count.as_fraction() == Fraction(1, 49)


def test_result_preserves_source() -> None:
    result = compute_gowers_cube_profile(5, (0, 1, 2), 2)
    assert result.modulus == 5
    assert result.subset == (0, 1, 2)
    assert result.order == 2


def test_noncanonical_subset_representative_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="canonical residues"):
        compute_gowers_cube_profile(5, (5,), 1)


def test_coupled_cube_work_is_rejected_before_enumeration() -> None:
    with pytest.raises(OperationDomainValidationError, match="vertex-check bound"):
        compute_gowers_cube_profile(3, (0, 1, 2), 12)


@pytest.mark.parametrize(
    ("modulus", "subset", "order"),
    [(2, (0, 1), 3), (3, (0, 1), 2), (4, (0, 1, 3), 2)],
)
def test_cube_count_matches_direct_vertex_definition(
    modulus: int, subset: tuple[int, ...], order: int
) -> None:
    result = compute_gowers_cube_profile(modulus, subset, order)

    assert result.cube_count == _direct_cube_count(modulus, subset, order)


@pytest.mark.scale
def test_full_binary_set_reaches_recurrence_vertex_boundary() -> None:
    result = compute_gowers_cube_profile(2, (0, 1), 9)

    assert result.cube_count == 2**10
    assert result.normalized_count.as_fraction() == 1

    with pytest.raises(OperationDomainValidationError, match="vertex-check bound"):
        compute_gowers_cube_profile(2, (0, 1), 10)


def test_serialized_forged_profile_is_rejected_by_verifier() -> None:
    result = compute_gowers_cube_profile(3, (0, 1), 1)
    payload = result.model_dump(mode="json")
    payload["cube_count"] += 1
    decoded = result.model_validate(payload)
    assert not verify_gowers_cube_profile(decoded)
