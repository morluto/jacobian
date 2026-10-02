"""Primitive commutator equations and bounded direct diagonal centralizers."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import pytest
from sympy import Matrix, eye, kronecker_product

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.canonical_forms import centralizer_basis
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


def _matrix(rows: Sequence[Sequence[int | Fraction]]) -> RationalMatrix:
    return rational_matrix_from_fractions(
        [[Fraction(value) for value in row] for row in rows]
    )


def _entries(matrix: RationalMatrix) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(value.as_fraction() for value in row) for row in matrix.entries)


@pytest.mark.parametrize(
    ("size", "scalar", "shift", "permuted"),
    [
        (3, Fraction(10**63), Fraction(0), False),
        (4, Fraction(-(10**127)), Fraction(10**255), True),
        (9, Fraction(10**255), Fraction(-(10**255)), False),
        (16, Fraction(10**255), Fraction(0), True),
        (16, Fraction(-(10**255), 10**255 - 1), Fraction(0), False),
        (3, Fraction(1, 10**255 - 1), Fraction(10**255), True),
    ],
)
def test_scaled_shifted_permuted_jordan_has_complete_unit_basis(
    size: int, scalar: Fraction, shift: Fraction, permuted: bool
) -> None:
    order = list(reversed(range(size))) if permuted else list(range(size))
    source = _matrix(
        [
            [shift if i == j else scalar if j == i + 1 else 0 for j in order]
            for i in order
        ]
    )
    result = centralizer_basis(source)
    # The Jordan recurrence makes every commuter upper Toeplitz before the
    # permutation: its complete basis is I,J,...,J^(n-1), with disjoint supports.
    expected = {
        tuple(tuple(Fraction(j == i + power) for j in order) for i in order)
        for power in range(size)
    }
    assert result.dimension == size
    assert {_entries(matrix) for matrix in result.basis} == expected
    a = Matrix(_entries(source))
    assert all(a * Matrix(_entries(b)) == Matrix(_entries(b)) * a for b in result.basis)


@pytest.mark.parametrize(
    "rows",
    [
        [[0, Fraction(3, 5), 0], [Fraction(2, 7), 0, Fraction(4, 11)], [0, 0, 0]],
        [[Fraction(1, 7), 1, 2], [3, Fraction(-2, 11), 4], [5, 6, Fraction(3, 13)]],
        [[2, 0, 1, 0], [0, 2, 0, 1], [0, 0, 3, 0], [0, 0, 0, 3]],
    ],
)
def test_primitive_rows_preserve_independent_commutator_rank(
    rows: list[list[int | Fraction]],
) -> None:
    source = _matrix(rows)
    result = centralizer_basis(source)
    a = Matrix(rows)
    n = a.rows
    basis = [Matrix(_entries(matrix)) for matrix in result.basis]
    # Independent column-major Kronecker construction, unlike the producer.
    commutator = kronecker_product(eye(n), a) - kronecker_product(a.T, eye(n))
    assert result.dimension == n * n - commutator.rank()
    assert (
        Matrix.hstack(*(b.reshape(n * n, 1) for b in basis)).rank() == result.dimension
    )
    assert all(a * b == b * a for b in basis)


@pytest.mark.parametrize(
    "diagonal",
    [[0], [0] * 3, list(range(17)), list(range(32)), [7] * 17, [0] * 8 + [1] * 9],
)
def test_larger_diagonal_basis_is_exactly_equal_eigenvalue_matrix_units(
    diagonal: list[int],
) -> None:
    n = len(diagonal)
    source = _matrix(
        [[diagonal[i] if i == j else 0 for j in range(n)] for i in range(n)]
    )
    result = centralizer_basis(source)
    supports = []
    for matrix in result.basis:
        support = [
            (i, j)
            for i, row in enumerate(matrix.entries)
            for j, value in enumerate(row)
            if value.num
        ]
        assert len(support) == 1
        i, j = support[0]
        assert matrix.entries[i][j].as_fraction() == 1
        supports.append((i, j))
    # [D,E_ij]=(d_i-d_j)E_ij establishes commutation and completeness;
    # distinct singleton supports establish linear independence.
    expected = [
        (i, j) for i in range(n) for j in range(n) if diagonal[i] == diagonal[j]
    ]
    assert result.dimension == len(expected)
    assert supports == expected


@pytest.mark.parametrize(("size", "scalar"), [(22, True), (64, False)])
def test_direct_basis_near_allocation_bound(size: int, scalar: bool) -> None:
    source = _matrix(
        [
            [
                int(scalar) if scalar and i == j else i if i == j else 0
                for j in range(size)
            ]
            for i in range(size)
        ]
    )
    result = centralizer_basis(source)
    assert result.dimension == (size * size if scalar else size)
    assert sum(len(row) for matrix in result.basis for row in matrix.entries) <= 262_144


@pytest.mark.parametrize(("size", "scalar"), [(23, True), (64, True), (65, False)])
def test_genuinely_oversized_direct_basis_is_resource_refused(
    size: int, scalar: bool
) -> None:
    source = _matrix(
        [
            [
                int(scalar) if scalar and i == j else i if i == j else 0
                for j in range(size)
            ]
            for i in range(size)
        ]
    )
    with pytest.raises(OperationResourceAdmissionError) as exc:
        centralizer_basis(source)
    assert exc.value.errors()[0]["type"] == "matrix.centralizer.output"


def test_general_system_order_bound_remains_resource_refusal() -> None:
    n = 17
    source = _matrix([[int(j == i + 1) for j in range(n)] for i in range(n)])
    with pytest.raises(OperationResourceAdmissionError) as exc:
        centralizer_basis(source)
    assert exc.value.errors()[0]["type"] == "matrix.centralizer.work"


def test_denominator_clearing_has_separate_presolve_bound() -> None:
    # Pairwise-coprime 250-digit denominators make the primitive rows large.
    from sympy import nextprime

    primes = [int(nextprime(10**249 + i * 10_000)) for i in range(16)]
    source = _matrix(
        [[Fraction(1, primes[(i + j) % 16]) for j in range(16)] for i in range(16)]
    )
    with pytest.raises(OperationResourceAdmissionError) as exc:
        centralizer_basis(source)
    assert exc.value.errors()[0]["type"] == "matrix.centralizer.presolve"


@pytest.mark.parametrize("source", [None, 1, [], {}])
def test_native_centralizer_requires_typed_matrix(source: Any) -> None:
    with pytest.raises(OperationDomainValidationError):
        centralizer_basis(source)


@pytest.mark.parametrize("rows", [[], [[1, 2]]])
def test_empty_and_nonsquare_centralizers_are_domain_errors(
    rows: list[list[int]],
) -> None:
    with pytest.raises(OperationDomainValidationError) as exc:
        centralizer_basis(_matrix(rows))
    assert exc.value.errors()[0]["type"] == "matrix.shape_mismatch"


def test_former_maximum_height_refusal_is_a_cheap_scaled_cyclic_shift() -> None:
    n = 16
    scalar = 10**255
    source = _matrix(
        [
            [
                scalar if i == j else scalar - 1 if j == (i + 1) % n else 0
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    result = centralizer_basis(source)
    # A single cyclic shift is cyclic over QQ; its n powers have disjoint
    # wraparound diagonals and form its complete centralizer.
    expected = {
        tuple(tuple(Fraction(j == (i + power) % n) for j in range(n)) for i in range(n))
        for power in range(n)
    }
    assert result.dimension == n
    assert {_entries(matrix) for matrix in result.basis} == expected
