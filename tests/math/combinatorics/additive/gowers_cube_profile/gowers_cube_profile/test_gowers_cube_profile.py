from __future__ import annotations

from jacobian.math.combinatorics.additive.gowers_cube_profile.operations import (
    compute_gowers_cube_profile,
)


def test_full_set() -> None:
    """Full set of Z/5Z should have maximum cube count."""
    result = compute_gowers_cube_profile(5, (0, 1, 2, 3, 4), 1)
    # Order 1: cubes are (x, x+e) with both in A. Full set -> all 5*5 = 25
    assert result.cube_count == 25


def test_empty_set() -> None:
    result = compute_gowers_cube_profile(5, (), 1)
    assert result.cube_count == 0


def test_single_element() -> None:
    result = compute_gowers_cube_profile(7, (3,), 1)
    # Only cube with both vertices at 3: base=3, e=0 -> (3,3)
    assert result.cube_count == 1


def test_result_preserves_source() -> None:
    result = compute_gowers_cube_profile(5, (0, 1, 2), 2)
    assert result.modulus == 5
    assert result.subset == (0, 1, 2)
    assert result.order == 2
