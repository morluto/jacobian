from __future__ import annotations

import pytest

from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    column_basis,
    nullspace,
    quotient_basis,
    rank,
    rref,
)


def test_rank_rref_and_nullspace_bind_the_prime() -> None:
    matrix = PrimeFieldMatrix(
        prime=2,
        entries=((1, 1, 0), (0, 1, 1)),
        columns=3,
    )

    assert rank(matrix) == 2
    assert rref(matrix) == (((1, 0, 1), (0, 1, 1)), (0, 1))
    assert nullspace(matrix) == ((1, 1, 1),)


def test_column_and_quotient_bases_are_source_ordered() -> None:
    matrix = PrimeFieldMatrix(
        prime=3,
        entries=((1, 2, 0), (0, 0, 1)),
        columns=3,
    )

    assert column_basis(matrix) == ((1, 0), (0, 1))
    assert quotient_basis(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0),),
        prime=3,
    ) == ((0, 1, 0), (0, 0, 1))


def test_empty_shapes_remain_explicit() -> None:
    zero_by_three = PrimeFieldMatrix(prime=2, entries=(), columns=3)
    three_by_zero = PrimeFieldMatrix(prime=2, entries=((), (), ()), columns=0)

    assert rank(zero_by_three) == 0
    assert nullspace(zero_by_three) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert rref(three_by_zero) == (((), (), ()), ())


def test_input_rejects_nonprime_or_ragged_semantics() -> None:
    with pytest.raises(ValueError, match="prime"):
        PrimeFieldMatrix(prime=4, entries=((1,),), columns=1)
    with pytest.raises(ValueError, match="column"):
        PrimeFieldMatrix(prime=2, entries=((1,),), columns=2)
