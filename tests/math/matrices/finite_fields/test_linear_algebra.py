from __future__ import annotations

from typing import Any, cast

import pytest

from jacobian.math.matrices.finite_fields import (
    linear_algebra as prime_field_linear_algebra,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
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


def test_column_basis_returns_original_pivot_columns() -> None:
    matrix = PrimeFieldMatrix(
        prime=5,
        entries=((1, 2, 3, 4), (0, 0, 1, 1)),
        columns=4,
    )
    assert column_basis(matrix) == ((1, 0), (3, 1))


def test_quotient_basis_keeps_original_cycle_vectors() -> None:
    cycles = ((2, 1, 0), (1, 1, 0), (0, 0, 1))
    assert quotient_basis(cycles, ((1, 3, 0),), prime=5) == ((1, 1, 0), (0, 0, 1))


def test_quotient_basis_rejects_ragged_vectors() -> None:
    with pytest.raises(ValueError, match="dimension"):
        quotient_basis(((1, 0),), ((1, 0, 0),), prime=3)


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


def test_matrix_rejects_noncanonical_entries() -> None:
    with pytest.raises(ValueError, match="canonical"):
        PrimeFieldMatrix(prime=2, entries=((3,),), columns=1)
    with pytest.raises(ValueError, match="canonical"):
        PrimeFieldMatrix(prime=3, entries=((1, -1),), columns=2)
    with pytest.raises(ValueError):
        PrimeFieldMatrix(prime=2, entries=cast(Any, ((1.0,),)), columns=1)


def test_matrix_accepts_canonical_residues_at_the_boundary() -> None:
    matrix = PrimeFieldMatrix(prime=5, entries=((0, 4), (4, 0)), columns=2)
    assert rank(matrix) == 2


def test_matrix_rejects_oversized_dimensions_before_primality() -> None:
    with pytest.raises(ValueError, match="dimension bound"):
        PrimeFieldMatrix(prime=2, entries=(), columns=257)
    with pytest.raises(ValueError, match="dimension bound"):
        PrimeFieldMatrix(prime=2, entries=((),) * 257, columns=0)


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the prime_field_linear_algebra public API."""
    expected = (
        "PrimeFieldMatrix",
        "column_basis",
        "nullspace",
        "quotient_basis",
        "rank",
        "rref",
    )
    assert tuple(prime_field_linear_algebra.__all__) == expected
    assert len(prime_field_linear_algebra.__all__) == len(
        set(prime_field_linear_algebra.__all__)
    )
    assert all(not name.startswith("_") for name in prime_field_linear_algebra.__all__)
    assert all(
        hasattr(prime_field_linear_algebra, name)
        for name in prime_field_linear_algebra.__all__
    )
