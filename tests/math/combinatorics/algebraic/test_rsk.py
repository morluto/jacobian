"""Permutation RSK native contract regression tests."""

import itertools

import pytest
from jacobian.math.combinatorics.algebraic import (
    FinitePermutation,
    inverse_permutation_rsk,
    permutation_rsk,
)


@pytest.mark.parametrize("size", range(7))
def test_permutation_rsk_round_trips_all_small_permutations(size: int) -> None:
    for images in itertools.permutations(range(1, size + 1)):
        permutation = FinitePermutation(images=images)
        pair = permutation_rsk(permutation)
        assert inverse_permutation_rsk(pair) == permutation
        assert pair.p_tableau.shape == pair.q_tableau.shape


def test_permutation_rsk_requires_canonical_finite_permutation() -> None:
    with pytest.raises(Exception, match="finite permutation"):
        permutation_rsk((1, 2, 3))

