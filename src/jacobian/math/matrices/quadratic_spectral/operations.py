"""Exact bounded matrix spectra over one real quadratic field."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from typing import Any, Literal

from jacobian.canonical import format_canonical_integer
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.matrices.quadratic_spectral.values import (
    Definiteness,
    RealAlgebraicMultiplicity,
    RealQuadraticInertia,
    RealQuadraticSpectrum,
    SpectrumKind,
)
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.real_algebraic import RealAlgebraicValue

type Quadratic = tuple[Fraction, Fraction]
type FractionPolynomial = tuple[Fraction, ...]
Branch = Literal["UPPER", "LOWER", "REPEATED"]

MAX_INERTIA_DIMENSION = 4
# A degree-eight factor of a degree-eight primitive norm polynomial with
# 996-digit coefficients has coefficients below 10**999 by the Landau--
# Mignotte factor bound.  Mignotte's separation bound then places the required
# isolating and Arb branch-selection precision below 32,768 bits; that fixed
# precision is therefore part of the admitted exact-kernel envelope, not a
# fallback for an otherwise unbounded numerical search.
MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS = 996
_MAX_ARB_PRECISION_BITS = 32_768
_ZERO: Quadratic = (Fraction(0), Fraction(0))
_ONE: Quadratic = (Fraction(1), Fraction(0))


@dataclass(frozen=True, slots=True)
class _RootData:
    value: RealAlgebraicValue
    lower: Any
    upper: Any


def _entry(value) -> Quadratic:  # type: ignore[no-untyped-def]
    return (
        value.rational_part.as_fraction(),
        value.radical_coefficient.as_fraction(),
    )


def _add(left: Quadratic, right: Quadratic) -> Quadratic:
    return left[0] + right[0], left[1] + right[1]


def _negate(value: Quadratic) -> Quadratic:
    return -value[0], -value[1]


def _subtract(left: Quadratic, right: Quadratic) -> Quadratic:
    return _add(left, _negate(right))


def _multiply(left: Quadratic, right: Quadratic, radicand: int) -> Quadratic:
    return (
        left[0] * right[0] + radicand * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _scale(value: Quadratic, scalar: Fraction) -> Quadratic:
    return value[0] * scalar, value[1] * scalar


def _is_zero(value: Quadratic) -> bool:
    return value == _ZERO


def _sign(value: Quadratic, radicand: int) -> int:
    rational, radical = value
    if radical == 0:
        return (rational > 0) - (rational < 0)
    if rational == 0:
        return (radical > 0) - (radical < 0)
    if (rational > 0) == (radical > 0):
        return (rational > 0) - (rational < 0)
    rational_square = rational * rational
    radical_square = radical * radical * radicand
    if rational_square == radical_square:  # impossible for square-free d > 1
        raise RuntimeError("quadratic-field sign comparison reached an invalid tie")
    dominant = radical if radical_square > rational_square else rational
    return (dominant > 0) - (dominant < 0)


def _divide(left: Quadratic, right: Quadratic, radicand: int) -> Quadratic:
    norm = right[0] * right[0] - radicand * right[1] * right[1]
    if norm == 0:
        raise ZeroDivisionError("cannot divide by zero in a real quadratic field")
    inverse = (right[0] / norm, -right[1] / norm)
    return _multiply(left, inverse, radicand)


def _convolve(left: FractionPolynomial, right: FractionPolynomial) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _primitive_integer_coefficients(
    coefficients_increasing: list[Fraction] | FractionPolynomial,
) -> tuple[int, ...]:
    denominator_lcm = 1
    for coefficient in coefficients_increasing:
        denominator_lcm = lcm(denominator_lcm, coefficient.denominator)
    integers: list[int] = [
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


def _matrix_entries(matrix: RealQuadraticMatrix) -> list[list[Quadratic]]:
    return [[_entry(value) for value in row] for row in matrix.entries]


def _trace_and_determinant(
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> tuple[Quadratic, Quadratic]:
    entries = _matrix_entries(matrix)
    radicand = matrix.entries[0][0].radicand
    if spectrum_kind == "SYMMETRIC_EIGENVALUES":
        a, b = entries[0]
        _ignored, c = entries[1]
    else:
        m00, m01 = entries[0]
        m10, m11 = entries[1]
        a = _add(
            _multiply(m00, m00, radicand),
            _multiply(m10, m10, radicand),
        )
        b = _add(
            _multiply(m00, m01, radicand),
            _multiply(m10, m11, radicand),
        )
        c = _add(
            _multiply(m01, m01, radicand),
            _multiply(m11, m11, radicand),
        )
    trace = _add(a, c)
    determinant = _subtract(
        _multiply(a, c, radicand),
        _multiply(b, b, radicand),
    )
    return trace, determinant


def _polynomial_parts(
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> tuple[FractionPolynomial, FractionPolynomial, Quadratic, Quadratic]:
    trace, determinant = _trace_and_determinant(matrix, spectrum_kind)
    if spectrum_kind == "SYMMETRIC_EIGENVALUES":
        rational: FractionPolynomial = (determinant[0], -trace[0], Fraction(1))
        radical: FractionPolynomial = (determinant[1], -trace[1], Fraction(0))
    else:
        rational = (
            determinant[0],
            Fraction(0),
            -trace[0],
            Fraction(0),
            Fraction(1),
        )
        radical = (
            determinant[1],
            Fraction(0),
            -trace[1],
            Fraction(0),
            Fraction(0),
        )
    return rational, radical, trace, determinant


def _annihilating_coefficients(
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> tuple[int, ...]:
    rational, radical, _trace, _determinant = _polynomial_parts(matrix, spectrum_kind)
    radicand = matrix.entries[0][0].radicand
    norm = [
        left - radicand * right
        for left, right in zip(
            _convolve(rational, rational),
            _convolve(radical, radical),
            strict=True,
        )
    ]
    return _primitive_integer_coefficients(norm)


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
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> None:
    coefficients = _annihilating_coefficients(matrix, spectrum_kind)
    if any(
        len(format_canonical_integer(coefficient).lstrip("-"))
        > MAX_SPECTRAL_ANNIHILATING_COEFFICIENT_DIGITS
        for coefficient in coefficients
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


def _sympy_polynomial(coefficients: tuple[int, ...]):  # type: ignore[no-untyped-def]
    import sympy

    return sympy.Poly.from_list(coefficients, gens=sympy.Symbol("x"), domain=sympy.ZZ)


def _canonical_factor(factor) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    coefficients = [int(coefficient) for coefficient in factor.all_coeffs()]
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    coefficients = [coefficient // content for coefficient in coefficients]
    if coefficients[0] < 0:
        coefficients = [-coefficient for coefficient in coefficients]
    return tuple(format_canonical_integer(coefficient) for coefficient in coefficients)


def _root_data(polynomial) -> tuple[_RootData, ...]:  # type: ignore[no-untyped-def]
    factors = polynomial.factor_list()[1]
    factor_indices = [0] * len(factors)
    rows: list[_RootData] = []
    for (lower, upper), _multiplicity in polynomial.intervals():
        matches = [
            index
            for index, (factor, _factor_multiplicity) in enumerate(factors)
            if strict_root_count(factor, lower, upper) == 1
        ]
        if len(matches) != 1:  # pragma: no cover
            raise RuntimeError("exact factor isolation did not identify one root")
        factor_index = matches[0]
        factor, _factor_multiplicity = factors[factor_index]
        root_index = factor_indices[factor_index]
        factor_indices[factor_index] += 1
        rows.append(
            _RootData(
                value=RealAlgebraicValue(
                    polynomial=_canonical_factor(factor),
                    real_root_index=root_index,
                ),
                lower=lower,
                upper=upper,
            )
        )
    return tuple(rows)


def _evaluate_polynomial(
    coefficients: FractionPolynomial,
    value: Fraction,
) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _rational_branches(
    roots: tuple[_RootData, ...],
    rational: FractionPolynomial,
    radical: FractionPolynomial,
    trace: Quadratic,
    spectrum_kind: SpectrumKind,
    radicand: int,
) -> dict[Branch, _RootData]:
    result: dict[Branch, _RootData] = {}
    for root in roots:
        if root.lower != root.upper:
            continue
        value = Fraction(int(root.lower.p), int(root.lower.q))
        if spectrum_kind == "SINGULAR_VALUES" and value < 0:
            continue
        if _evaluate_polynomial(rational, value) != 0:
            continue
        if _evaluate_polynomial(radical, value) != 0:
            continue
        branch_difference = _subtract(
            (2 * value, Fraction(0))
            if spectrum_kind == "SYMMETRIC_EIGENVALUES"
            else (2 * value * value, Fraction(0)),
            trace,
        )
        sign = _sign(branch_difference, radicand)
        branch: Branch = "UPPER" if sign > 0 else "LOWER" if sign < 0 else "REPEATED"
        result[branch] = root
    return result


def _arb_fraction(value: Fraction):  # type: ignore[no-untyped-def]
    from flint import arb

    return arb(value.numerator) / value.denominator


def _arb_quadratic(value: Quadratic, radicand: int):  # type: ignore[no-untyped-def]
    from flint import arb

    return _arb_fraction(value[0]) + _arb_fraction(value[1]) * arb(radicand).sqrt()


def _arb_rational(value: Any) -> Any:
    from flint import arb

    return arb(int(value.p)) / int(value.q)


def _strictly_inside(ball, root: _RootData) -> bool:  # type: ignore[no-untyped-def]
    if root.lower == root.upper:
        return False
    return bool(ball > _arb_rational(root.lower) and ball < _arb_rational(root.upper))


def _branch_balls(
    trace: Quadratic,
    determinant: Quadratic,
    spectrum_kind: SpectrumKind,
    radicand: int,
    precision: int,
    repeated: bool,
) -> dict[Branch, Any] | None:
    from flint import ctx

    with ctx.workprec(precision):
        trace_ball = _arb_quadratic(trace, radicand)
        if repeated:
            repeated_ball = trace_ball / 2
            if spectrum_kind == "SINGULAR_VALUES":
                if not repeated_ball > 0:
                    return None
                repeated_ball = repeated_ball.sqrt()
            return {"REPEATED": repeated_ball}
        determinant_ball = _arb_quadratic(determinant, radicand)
        discriminant_ball = trace_ball * trace_ball - 4 * determinant_ball
        if not discriminant_ball > 0:
            return None
        root = discriminant_ball.sqrt()
        upper = (trace_ball + root) / 2
        lower = (trace_ball - root) / 2
        if spectrum_kind == "SINGULAR_VALUES":
            if not upper > 0:
                return None
            upper = upper.sqrt()
            lower = lower.sqrt() if lower > 0 else None
        result: dict[Branch, Any] = {"UPPER": upper}
        if lower is not None:
            result["LOWER"] = lower
        return result


def _select_nonrational_branches(
    roots: tuple[_RootData, ...],
    trace: Quadratic,
    determinant: Quadratic,
    spectrum_kind: SpectrumKind,
    radicand: int,
    missing: tuple[Branch, ...],
) -> dict[Branch, _RootData]:
    precision = 128
    while precision <= _MAX_ARB_PRECISION_BITS:
        balls = _branch_balls(
            trace,
            determinant,
            spectrum_kind,
            radicand,
            precision,
            repeated=missing == ("REPEATED",),
        )
        if balls is not None:
            selected: dict[Branch, _RootData] = {}
            for branch in missing:
                ball = balls.get(branch)
                if ball is None:
                    break
                matches = [root for root in roots if _strictly_inside(ball, root)]
                if len(matches) != 1:
                    break
                selected[branch] = matches[0]
            else:
                if len({row.value for row in selected.values()}) == len(selected):
                    return selected
        precision *= 2
    raise RuntimeError("rigorous algebraic-root selection exceeded its precision proof")


def spectrum_rows(
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> tuple[RealAlgebraicMultiplicity, ...]:
    """Return the complete descending exact spectrum with multiplicities."""

    if spectrum_kind == "SYMMETRIC_EIGENVALUES":
        require_symmetric_spectrum_matrix(matrix)
    else:
        require_singular_spectrum_matrix(matrix)
    rational, radical, trace, determinant = _polynomial_parts(matrix, spectrum_kind)
    radicand = matrix.entries[0][0].radicand
    discriminant = _subtract(
        _multiply(trace, trace, radicand),
        _scale(determinant, Fraction(4)),
    )
    discriminant_sign = _sign(discriminant, radicand)
    if discriminant_sign < 0:  # pragma: no cover
        raise RuntimeError("a real symmetric 2 by 2 spectrum had negative discriminant")

    polynomial = _sympy_polynomial(_annihilating_coefficients(matrix, spectrum_kind))
    roots = _root_data(polynomial)
    rational_roots = _rational_branches(
        roots, rational, radical, trace, spectrum_kind, radicand
    )
    if discriminant_sign == 0:
        branches: tuple[tuple[Branch, int], ...] = (("REPEATED", 2),)
    else:
        branches = (("UPPER", 1), ("LOWER", 1))

    selected = dict(rational_roots)
    missing = tuple(
        branch for branch, _multiplicity in branches if branch not in selected
    )
    if missing:
        selected.update(
            _select_nonrational_branches(
                roots,
                trace,
                determinant,
                spectrum_kind,
                radicand,
                missing,
            )
        )
    return tuple(
        RealAlgebraicMultiplicity(
            value=selected[branch].value, multiplicity=multiplicity
        )
        for branch, multiplicity in branches
    )


def symmetric_spectrum(matrix: RealQuadraticMatrix) -> RealQuadraticSpectrum:
    """Return the exact descending eigenvalue spectrum of a symmetric 2 by 2 matrix."""

    return RealQuadraticSpectrum(
        matrix=matrix,
        spectrum_kind="SYMMETRIC_EIGENVALUES",
        values=spectrum_rows(matrix, "SYMMETRIC_EIGENVALUES"),
    )


def singular_spectrum(matrix: RealQuadraticMatrix) -> RealQuadraticSpectrum:
    """Return the exact descending singular-value spectrum of a 2 by 2 matrix."""

    return RealQuadraticSpectrum(
        matrix=matrix,
        spectrum_kind="SINGULAR_VALUES",
        values=spectrum_rows(matrix, "SINGULAR_VALUES"),
    )


def _swap_symmetric(matrix: list[list[Quadratic]], left: int, right: int) -> None:
    if left == right:
        return
    matrix[left], matrix[right] = matrix[right], matrix[left]
    for row in matrix:
        row[left], row[right] = row[right], row[left]


def _find_off_diagonal(
    matrix: list[list[Quadratic]], index: int
) -> tuple[int, int] | None:
    for row in range(index, len(matrix)):
        for column in range(row + 1, len(matrix)):
            if not _is_zero(matrix[row][column]):
                return row, column
    return None


def _two_by_two_inertia(
    aa: Quadratic,
    bb: Quadratic,
    cc: Quadratic,
    radicand: int,
) -> tuple[int, int, int]:
    determinant = _subtract(
        _multiply(aa, cc, radicand),
        _multiply(bb, bb, radicand),
    )
    determinant_sign = _sign(determinant, radicand)
    if determinant_sign < 0:
        return 1, 1, 0
    trace_sign = _sign(_add(aa, cc), radicand)
    if determinant_sign > 0:
        return (2, 0, 0) if trace_sign > 0 else (0, 2, 0)
    if trace_sign > 0:
        return 1, 0, 1
    if trace_sign < 0:
        return 0, 1, 1
    return 0, 0, 2


def _eliminate_one(
    matrix: list[list[Quadratic]],
    index: int,
    pivot: int,
    radicand: int,
) -> int:
    _swap_symmetric(matrix, index, pivot)
    diagonal = matrix[index][index]
    for row in range(index + 1, len(matrix)):
        if _is_zero(matrix[row][index]):
            continue
        factor = _divide(matrix[row][index], diagonal, radicand)
        for column in range(index, len(matrix)):
            matrix[row][column] = _subtract(
                matrix[row][column],
                _multiply(factor, matrix[index][column], radicand),
            )
        for column in range(index, len(matrix)):
            matrix[column][row] = matrix[row][column]
    return _sign(diagonal, radicand)


def _eliminate_two(
    matrix: list[list[Quadratic]],
    index: int,
    radicand: int,
) -> tuple[int, int, int]:
    off_diagonal = _find_off_diagonal(matrix, index)
    if off_diagonal is None:  # pragma: no cover
        raise RuntimeError("2 by 2 pivot requested without an off-diagonal entry")
    first, second = off_diagonal
    _swap_symmetric(matrix, index, first)
    if second == index:
        second = first
    _swap_symmetric(matrix, index + 1, second)
    aa = matrix[index][index]
    bb = matrix[index][index + 1]
    cc = matrix[index + 1][index + 1]
    counts = _two_by_two_inertia(aa, bb, cc, radicand)
    determinant = _subtract(
        _multiply(aa, cc, radicand),
        _multiply(bb, bb, radicand),
    )
    if index + 2 < len(matrix):
        inverse_00 = _divide(cc, determinant, radicand)
        inverse_01 = _divide(_negate(bb), determinant, radicand)
        inverse_11 = _divide(aa, determinant, radicand)
        for row in range(index + 2, len(matrix)):
            left = matrix[row][index]
            right = matrix[row][index + 1]
            coefficient_0 = _add(
                _multiply(left, inverse_00, radicand),
                _multiply(right, inverse_01, radicand),
            )
            coefficient_1 = _add(
                _multiply(left, inverse_01, radicand),
                _multiply(right, inverse_11, radicand),
            )
            for column in range(index, len(matrix)):
                matrix[row][column] = _subtract(
                    matrix[row][column],
                    _add(
                        _multiply(coefficient_0, matrix[index][column], radicand),
                        _multiply(
                            coefficient_1,
                            matrix[index + 1][column],
                            radicand,
                        ),
                    ),
                )
            for column in range(index, len(matrix)):
                matrix[column][row] = matrix[row][column]
    return counts


def _inertia_counts(matrix: RealQuadraticMatrix) -> tuple[int, int, int]:
    radicand = matrix.entries[0][0].radicand
    reduced = _matrix_entries(matrix)
    positive = negative = zero = 0
    index = 0
    while index < len(reduced):
        pivot = next(
            (
                row
                for row in range(index, len(reduced))
                if not _is_zero(reduced[row][row])
            ),
            None,
        )
        if pivot is not None:
            sign = _eliminate_one(reduced, index, pivot, radicand)
            positive += sign > 0
            negative += sign < 0
            index += 1
            continue
        if _find_off_diagonal(reduced, index) is None:
            zero += len(reduced) - index
            break
        row_positive, row_negative, row_zero = _eliminate_two(reduced, index, radicand)
        positive += row_positive
        negative += row_negative
        zero += row_zero
        index += 2
    return positive, negative, zero


def _definiteness(positive: int, negative: int, zero: int) -> Definiteness:
    if positive == 0 and negative == 0:
        return "zero"
    if zero == 0:
        if negative == 0:
            return "positive_definite"
        if positive == 0:
            return "negative_definite"
        return "indefinite"
    if negative == 0:
        return "positive_semidefinite"
    if positive == 0:
        return "negative_semidefinite"
    return "indefinite"


def inertia_data(
    matrix: RealQuadraticMatrix,
) -> tuple[int, int, int, Definiteness]:
    """Return exact Sylvester inertia data for an admitted matrix."""

    require_inertia_matrix(matrix)
    positive, negative, zero = _inertia_counts(matrix)
    return positive, negative, zero, _definiteness(positive, negative, zero)


def inertia(matrix: RealQuadraticMatrix) -> RealQuadraticInertia:
    """Return exact Sylvester inertia for a symmetric real-quadratic matrix."""

    positive, negative, zero, definiteness = inertia_data(matrix)
    return RealQuadraticInertia(
        matrix=matrix,
        n_positive=positive,
        n_negative=negative,
        n_zero=zero,
        definiteness=definiteness,
    )


__all__ = ["inertia", "singular_spectrum", "symmetric_spectrum"]
