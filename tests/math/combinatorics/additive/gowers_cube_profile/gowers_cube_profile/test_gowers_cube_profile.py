from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.gowers_cube_profile.operations import (
    compute_gowers_cube_profile,
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
