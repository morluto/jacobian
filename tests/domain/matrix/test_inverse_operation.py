"""Public matrix-inverse operation tests.

These tests invoke the catalog declaration rather than importing the private
Smith-normal-form kernel.  The kernel is exercised through the public
``matrix.inverse.compute`` contract, while this module asserts the observable
rational inverse and request-boundary behavior.
"""

from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from jacobian.contracts.matrix_operations import MatrixInverseResult
from jacobian.serving_catalog import ServingCatalog

_CATALOG = ServingCatalog.open()


def _run_inverse(entries: list[list[str]]) -> MatrixInverseResult:
    operation = _CATALOG.operation("matrix.inverse.compute")
    assert operation is not None
    request = operation.request_type.model_validate({"matrix": {"entries": entries}})
    result = operation.run(request)
    assert isinstance(result, MatrixInverseResult)
    return result


def _random_unimodular(random: Random, size: int) -> list[list[int]]:
    matrix = [[int(row == column) for column in range(size)] for row in range(size)]
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


def _multiply(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _fraction_entries(entries: list[list[str]]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(int(value)) for value in row) for row in entries)


def _result_entries(result: MatrixInverseResult) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value.as_fraction() for value in row) for row in result.inverse.entries
    )


def test_inverse_operation_returns_exact_two_sided_inverse() -> None:
    source = [["1", "1"], ["0", "1"]]
    result = _run_inverse(source)
    inverse = _result_entries(result)

    assert inverse == ((Fraction(1), Fraction(-1)), (Fraction(0), Fraction(1)))
    assert _multiply(_fraction_entries(source), inverse) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def test_inverse_operation_accepts_determinant_minus_one() -> None:
    source = [["0", "1"], ["1", "0"]]
    result = _run_inverse(source)
    inverse = _result_entries(result)

    assert inverse == ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
    assert _multiply(_fraction_entries(source), inverse) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def test_inverse_operation_returns_rational_values_for_non_unimodular_matrix() -> None:
    result = _run_inverse([["2", "0"], ["0", "1"]])

    assert _result_entries(result) == (
        (Fraction(1, 2), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_inverse_operation_round_trips_random_unimodular_matrices(size: int) -> None:
    random = Random(1127 + size)
    for _ in range(20):
        source = _random_unimodular(random, size)
        source_wire = [[str(value) for value in row] for row in source]
        inverse = _result_entries(_run_inverse(source_wire))
        source_fraction = tuple(
            tuple(Fraction(value) for value in row) for row in source
        )
        identity = tuple(
            tuple(Fraction(int(row == column)) for column in range(size))
            for row in range(size)
        )
        assert _multiply(source_fraction, inverse) == identity
        assert _multiply(inverse, source_fraction) == identity


def test_inverse_operation_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _run_inverse([])


def test_inverse_operation_rejects_non_square_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        _run_inverse([["1", "0"]])


def test_inverse_operation_rejects_singular_matrices() -> None:
    with pytest.raises(ValueError, match="singular"):
        _run_inverse([["1", "2"], ["2", "4"]])
