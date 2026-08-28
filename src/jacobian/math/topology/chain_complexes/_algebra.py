"""Exact linear-algebra primitives shared by admission and operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.topology.chain_complexes.values import ChainComplexValue


def parse_matrix(
    matrix: tuple[tuple[str, ...], ...], rows: int, cols: int, prime: int | None
) -> list[list[Fraction]]:
    """Parse an already-admitted canonical matrix into exact coefficients."""
    result = [[Fraction(0)] * cols for _ in range(rows)]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if prime is not None:
                result[i][j] = Fraction(int(value) % prime)
            elif "/" in value:
                numerator, denominator = value.split("/", 1)
                result[i][j] = Fraction(int(numerator), int(denominator))
            else:
                result[i][j] = Fraction(int(value))
    return result


def parsed_differentials(
    complex_value: ChainComplexValue,
) -> list[list[list[Fraction]]]:
    """Parse a complex's differentials while retaining declared widths."""
    return [
        parse_matrix(
            matrix,
            complex_value.basis_sizes[index],
            complex_value.basis_sizes[index + 1],
            complex_value.prime,
        )
        for index, matrix in enumerate(complex_value.differential_matrices)
    ]


def matrix_multiply(
    left: list[list[Fraction]],
    right: list[list[Fraction]],
    *,
    prime: int | None,
    result_columns: int | None = None,
    left_declared_columns: int | None = None,
) -> list[list[Fraction]]:
    """Multiply exact matrices without losing zero-width shape information."""
    left_columns = len(left[0]) if left else (left_declared_columns or 0)
    if right:
        right_columns = len(right[0])
        if len(right) != left_columns:
            raise ValueError("inner product dimensions do not match")
    else:
        right_columns = result_columns if result_columns is not None else 0
        if left_columns:
            raise ValueError("inner product dimensions do not match")
    result = [[Fraction(0)] * right_columns for _ in range(len(left))]
    for row_index in range(len(left)):
        for column_index in range(right_columns):
            value = sum(
                (
                    left[row_index][index] * right[index][column_index]
                    for index in range(left_columns)
                ),
                Fraction(0),
            )
            if prime is not None:
                value = Fraction(
                    value.numerator * pow(value.denominator, -1, prime) % prime
                )
            result[row_index][column_index] = value
    return result


def require_square_zero(complex_value: ChainComplexValue, *, label: str) -> None:
    """Raise when adjacent differentials of a candidate do not compose to zero."""
    differentials = parsed_differentials(complex_value)
    for index in range(len(differentials) - 1):
        product = matrix_multiply(
            differentials[index],
            differentials[index + 1],
            prime=complex_value.prime,
            left_declared_columns=complex_value.basis_sizes[index + 1],
            result_columns=complex_value.basis_sizes[index + 2],
        )
        if any(value != 0 for row in product for value in row):
            raise ValueError(
                f"{label} complex violates d^2=0 at chain degree "
                f"{complex_value.degree_min + index + 1}"
            )


def require_chain_map_relation(
    source: ChainComplexValue,
    target: ChainComplexValue,
    map_matrices: tuple[tuple[tuple[str, ...], ...], ...],
) -> None:
    """Raise when an already-shaped candidate fails the chain-map equation."""
    source_diffs = parsed_differentials(source)
    target_diffs = parsed_differentials(target)
    maps = [
        parse_matrix(
            matrix, target.basis_sizes[index], source.basis_sizes[index], source.prime
        )
        for index, matrix in enumerate(map_matrices)
    ]
    for index in range(len(source_diffs)):
        left = matrix_multiply(
            target_diffs[index],
            maps[index + 1],
            prime=source.prime,
            left_declared_columns=target.basis_sizes[index + 1],
            result_columns=source.basis_sizes[index + 1],
        )
        right = matrix_multiply(
            maps[index],
            source_diffs[index],
            prime=source.prime,
            left_declared_columns=source.basis_sizes[index],
            result_columns=source.basis_sizes[index + 1],
        )
        if left != right:
            raise ValueError(
                f"chain map does not commute with differentials at degree index {index}"
            )
