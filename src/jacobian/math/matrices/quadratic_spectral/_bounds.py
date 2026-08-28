"""Pure admission helpers for exact real-quadratic matrix spectra."""

from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm

from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.quadratic_spectral.values import SpectrumKind
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue

type Quadratic = tuple[Fraction, Fraction]
type FractionPolynomial = tuple[Fraction, ...]

MAX_INERTIA_DIMENSION = 4
MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS = 996


def _entry(value: RealQuadraticValue) -> Quadratic:
    return (
        value.rational_part.as_fraction(),
        value.radical_coefficient.as_fraction(),
    )


def _add(left: Quadratic, right: Quadratic) -> Quadratic:
    return left[0] + right[0], left[1] + right[1]


def _subtract(left: Quadratic, right: Quadratic) -> Quadratic:
    return left[0] - right[0], left[1] - right[1]


def _multiply(left: Quadratic, right: Quadratic, radicand: int) -> Quadratic:
    return (
        left[0] * right[0] + radicand * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _convolve(left: FractionPolynomial, right: FractionPolynomial) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _primitive_integer_coefficients(
    coefficients_increasing: list[Fraction],
) -> tuple[int, ...]:
    denominator_lcm = 1
    for coefficient in coefficients_increasing:
        denominator_lcm = lcm(denominator_lcm, coefficient.denominator)
    integers = [
        coefficient.numerator * (denominator_lcm // coefficient.denominator)
        for coefficient in coefficients_increasing
    ]
    while len(integers) > 1 and integers[-1] == 0:
        integers.pop()
    content = 0
    for integer in integers:
        content = gcd(content, abs(integer))
    integers = [integer // content for integer in integers]
    if integers[-1] < 0:
        integers = [-coefficient for coefficient in integers]
    return tuple(reversed(integers))


def annihilating_coefficients(
    matrix: RealQuadraticMatrix, spectrum_kind: SpectrumKind
) -> tuple[int, ...]:
    entries = [[_entry(value) for value in row] for row in matrix.entries]
    radicand = matrix.entries[0][0].radicand
    if spectrum_kind == "SYMMETRIC_EIGENVALUES":
        a, b = entries[0]
        _ignored, c = entries[1]
    else:
        m00, m01 = entries[0]
        m10, m11 = entries[1]
        a = _add(_multiply(m00, m00, radicand), _multiply(m10, m10, radicand))
        b = _add(_multiply(m00, m01, radicand), _multiply(m10, m11, radicand))
        c = _add(_multiply(m01, m01, radicand), _multiply(m11, m11, radicand))
    trace = _add(a, c)
    determinant = _subtract(_multiply(a, c, radicand), _multiply(b, b, radicand))
    if spectrum_kind == "SYMMETRIC_EIGENVALUES":
        rational: FractionPolynomial = (determinant[0], -trace[0], Fraction(1))
        radical: FractionPolynomial = (determinant[1], -trace[1], Fraction(0))
    else:
        rational = (determinant[0], Fraction(0), -trace[0], Fraction(0), Fraction(1))
        radical = (determinant[1], Fraction(0), -trace[1], Fraction(0), Fraction(0))
    return _primitive_integer_coefficients(
        [
            left - radicand * right
            for left, right in zip(
                _convolve(rational, rational),
                _convolve(radical, radical),
                strict=True,
            )
        ]
    )


def _require_two_by_two(matrix: RealQuadraticMatrix) -> None:
    if len(matrix.entries) != 2 or len(matrix.entries[0]) != 2:
        raise ValueError("exact quadratic spectral operations require a 2 by 2 matrix")


def _require_symmetric(matrix: RealQuadraticMatrix) -> None:
    rows = len(matrix.entries)
    columns = len(matrix.entries[0])
    if rows != columns:
        raise ValueError("quadratic inertia and eigenvalues require a square matrix")
    if any(
        matrix.entries[row][column] != matrix.entries[column][row]
        for row in range(rows)
        for column in range(row + 1, rows)
    ):
        raise ValueError("quadratic inertia and eigenvalues require exact symmetry")


def _require_spectral_coefficient_bound(
    matrix: RealQuadraticMatrix, spectrum_kind: SpectrumKind
) -> None:
    if any(
        len(format_canonical_integer(coefficient).lstrip("-"))
        > MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS
        for coefficient in annihilating_coefficients(matrix, spectrum_kind)
    ):
        raise ValueError(
            "exact spectral annihilating polynomial exceeds the "
            f"{MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS}-digit coefficient bound"
        )


def require_symmetric_spectrum_matrix(matrix: RealQuadraticMatrix) -> None:
    _require_two_by_two(matrix)
    _require_symmetric(matrix)
    _require_spectral_coefficient_bound(matrix, "SYMMETRIC_EIGENVALUES")


def require_singular_spectrum_matrix(matrix: RealQuadraticMatrix) -> None:
    _require_two_by_two(matrix)
    _require_spectral_coefficient_bound(matrix, "SINGULAR_VALUES")


def require_inertia_matrix(matrix: RealQuadraticMatrix) -> None:
    _require_symmetric(matrix)
    if len(matrix.entries) > MAX_INERTIA_DIMENSION:
        raise ValueError(
            f"exact quadratic inertia supports dimension at most {MAX_INERTIA_DIMENSION}"
        )


__all__ = [
    "MAX_INERTIA_DIMENSION",
    "MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS",
    "annihilating_coefficients",
    "require_inertia_matrix",
    "require_singular_spectrum_matrix",
    "require_symmetric_spectrum_matrix",
]
