from __future__ import annotations

from random import Random

import pytest

from jacobian.domains._certified_snf import (
    identity_matrix,
    inverse_unimodular,
    matrix_multiply,
)


def _random_unimodular(random: Random, size: int) -> list[list[int]]:
    matrix = identity_matrix(size)
    for _ in range(size * 6):
        i, j = random.randrange(size), random.randrange(size)
        if i == j:
            if random.random() < 0.5:
                matrix[i] = [-value for value in matrix[i]]
            continue
        factor = random.randint(-5, 5)
        matrix[i] = [
            value + factor * other
            for value, other in zip(matrix[i], matrix[j], strict=True)
        ]
    if random.random() < 0.5:
        a, b = random.randrange(size), random.randrange(size)
        matrix[a], matrix[b] = matrix[b], matrix[a]
    return matrix


def test_inverse_unimodular_empty_matrix() -> None:
    assert inverse_unimodular([]) == []


def test_inverse_unimodular_recovers_identity_on_sl2() -> None:
    source = [[1, 1], [0, 1]]
    inverse = inverse_unimodular(source)
    assert inverse == [[1, -1], [0, 1]]
    assert matrix_multiply(source, inverse) == identity_matrix(2)
    assert matrix_multiply(inverse, source) == identity_matrix(2)


def test_inverse_unimodular_accepts_determinant_minus_one() -> None:
    source = [[0, 1], [1, 0]]
    inverse = inverse_unimodular(source)
    assert matrix_multiply(source, inverse) == identity_matrix(2)


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_inverse_unimodular_round_trips_random_unimodular_matrices(size: int) -> None:
    random = Random(1127 + size)
    for _ in range(20):
        source = _random_unimodular(random, size)
        inverse = inverse_unimodular(source)
        assert matrix_multiply(source, inverse) == identity_matrix(size)
        assert matrix_multiply(inverse, source) == identity_matrix(size)


def test_inverse_unimodular_rejects_non_square_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        inverse_unimodular([[1, 0]])


def test_inverse_unimodular_rejects_singular_matrices() -> None:
    with pytest.raises(ValueError, match="singular"):
        inverse_unimodular([[1, 2], [2, 4]])


def test_inverse_unimodular_rejects_non_unimodular_matrices() -> None:
    with pytest.raises(ValueError, match="unimodular"):
        inverse_unimodular([[2, 0], [0, 1]])
