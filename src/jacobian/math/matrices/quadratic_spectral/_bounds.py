"""Pure admission helpers for exact real-quadratic matrix spectra."""

from __future__ import annotations

from fractions import Fraction
from math import factorial, gcd, lcm

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.quadratic_spectral.values import SpectrumKind
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue

type Quadratic = tuple[Fraction, Fraction]
type FractionPolynomial = tuple[Fraction, ...]

MAX_INERTIA_DIMENSION = 16
# Fraction-free congruence elimination can carry a denominator-cleared entry
# through several determinant minors.  This is a decimal-digit budget for the
# conservative preflight estimate below; diagonal forms use a direct sign
# path and do not pay this bound.
MAX_INERTIA_INTERMEDIATE_DIGITS = 200_000
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
    if len(matrix.entries) != 2 or not matrix.entries or len(matrix.entries[0]) != 2:
        raise ValueError("exact quadratic spectral operations require a 2 by 2 matrix")
    entries = [[_entry(value) for value in row] for row in matrix.entries]
    radicand = matrix.radicand
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
    if len(matrix.entries) != 2 or not matrix.entries or len(matrix.entries[0]) != 2:
        raise ValueError("exact quadratic spectral operations require a 2 by 2 matrix")


def _require_symmetric(matrix: RealQuadraticMatrix) -> None:
    rows = len(matrix.entries)
    if not matrix.entries:
        raise ValueError("quadratic matrix operations require a nonempty matrix")
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
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="matrix.inertia.dimension_bound",
            message=f"exact quadratic inertia supports dimension at most {MAX_INERTIA_DIMENSION}",
        )
    if all(
        matrix.entries[row][column].rational_part.num == 0
        and matrix.entries[row][column].radical_coefficient.num == 0
        for row in range(len(matrix.entries))
        for column in range(row + 1, len(matrix.entries))
    ):
        return
    order = len(matrix.entries)
    maximum_source_digits = max(
        len(format_canonical_integer(abs(component.num)))
        for row in matrix.entries
        for value in row
        for component in (value.rational_part, value.radical_coefficient)
    )
    maximum_scale_digits = 0
    for index in range(order):
        scale = lcm(
            *(
                component.den
                for other in range(order)
                for value in (matrix.entries[index][other],)
                for component in (value.rational_part, value.radical_coefficient)
            )
        )
        scale_digits = 0 if scale == 1 else len(format_canonical_integer(scale))
        maximum_scale_digits = max(maximum_scale_digits, scale_digits)
    cleared_entry_digits = maximum_source_digits + 2 * maximum_scale_digits
    radicand_digits = len(format_canonical_integer(matrix.radicand))
    # Bound both real embeddings of an entry, then each determinant by n!
    # products. Recovering either integral pair coefficient from the two
    # embeddings does not increase this bound (d >= 2). The block update
    # multiplies three minors; exact division multiplies once more by the
    # conjugate prior minor. Include the quadratic-product radicand factors.
    minor_digits = order * (
        cleared_entry_digits + (radicand_digits + 1) // 2 + 1
    ) + len(str(factorial(order)))
    estimated_digits = 4 * minor_digits + 4 * radicand_digits + 8
    if estimated_digits > MAX_INERTIA_INTERMEDIATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrix",),
            code="matrix.inertia.growth_bound",
            message="quadratic inertia intermediate integer growth exceeds the "
            f"{MAX_INERTIA_INTERMEDIATE_DIGITS}-digit bound",
        )


__all__ = [
    "MAX_INERTIA_DIMENSION",
    "MAX_INERTIA_INTERMEDIATE_DIGITS",
    "MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS",
    "annihilating_coefficients",
    "require_inertia_matrix",
    "require_singular_spectrum_matrix",
    "require_symmetric_spectrum_matrix",
]
