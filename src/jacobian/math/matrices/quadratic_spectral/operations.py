"""Exact bounded matrix spectra over one real quadratic field."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from typing import TYPE_CHECKING, Literal

from jacobian._flint import flint_workprec
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.matrices.quadratic_spectral._bounds import (
    annihilating_coefficients,
    require_inertia_matrix,
    require_singular_spectrum_matrix,
    require_symmetric_spectrum_matrix,
)
from jacobian.math.matrices.quadratic_spectral.values import (
    Definiteness,
    RealAlgebraicMultiplicity,
    RealQuadraticInertia,
    RealQuadraticSpectrum,
    SpectrumKind,
    _definiteness_from_inertia,
)
from jacobian.math.matrices.values import RealQuadraticMatrix
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticValue,
    require_square_free_radicand,
)
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue

if TYPE_CHECKING:
    from flint import arb
    from sympy import Poly
    from sympy.core.numbers import Rational as SympyRational

type Quadratic = tuple[Fraction, Fraction]
type IntegralQuadratic = tuple[int, int]
type QuadraticEntry = Quadratic | IntegralQuadratic
type FractionPolynomial = tuple[Fraction, ...]
Branch = Literal["UPPER", "LOWER", "REPEATED"]

# A degree-eight factor of a degree-eight primitive norm polynomial with
# 996-digit coefficients has coefficients below 10**999 by the Landau--
# Mignotte factor bound.  Mignotte's separation bound then places the required
# isolating and Arb branch-selection precision below 32,768 bits; that fixed
# precision is therefore part of the admitted exact-kernel envelope, not a
# fallback for an otherwise unbounded numerical search.
_MAX_ARB_PRECISION_BITS = 32_768
_ZERO: Quadratic = (Fraction(0), Fraction(0))
_ONE: Quadratic = (Fraction(1), Fraction(0))


@dataclass(frozen=True, slots=True)
class _RootData:
    value: RealAlgebraicValue
    lower: SympyRational
    upper: SympyRational


def _entry(value: RealQuadraticValue) -> Quadratic:
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


def _is_zero(value: QuadraticEntry) -> bool:
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


def _matrix_entries(matrix: RealQuadraticMatrix) -> list[list[Quadratic]]:
    return [[_entry(value) for value in row] for row in matrix.entries]


def _trace_and_determinant(
    matrix: RealQuadraticMatrix,
    spectrum_kind: SpectrumKind,
) -> tuple[Quadratic, Quadratic]:
    entries = _matrix_entries(matrix)
    radicand = matrix.radicand
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


def _sympy_polynomial(coefficients: tuple[int, ...]) -> Poly:
    import sympy

    return sympy.Poly.from_list(coefficients, gens=sympy.Symbol("x"), domain=sympy.ZZ)


def _canonical_factor(factor: Poly) -> tuple[int, ...]:
    coefficients = [int(coefficient) for coefficient in factor.all_coeffs()]
    content = 0
    for coefficient in coefficients:
        content = gcd(content, abs(coefficient))
    coefficients = [coefficient // content for coefficient in coefficients]
    if coefficients[0] < 0:
        coefficients = [-coefficient for coefficient in coefficients]
    return tuple(coefficients)


def _root_data(polynomial: Poly) -> tuple[_RootData, ...]:
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


def _arb_fraction(value: Fraction) -> arb:
    from flint import arb

    return arb(value.numerator) / value.denominator


def _arb_quadratic(value: Quadratic, radicand: int) -> arb:
    from flint import arb

    return _arb_fraction(value[0]) + _arb_fraction(value[1]) * arb(radicand).sqrt()


def _arb_rational(value: SympyRational) -> arb:
    from flint import arb

    return arb(int(value.p)) / int(value.q)


def _strictly_inside(ball: arb, root: _RootData) -> bool:
    if root.lower == root.upper:
        return False
    return bool(ball > _arb_rational(root.lower) and ball < _arb_rational(root.upper))


def _branch_balls(
    trace: Quadratic,
    determinant: Quadratic,
    spectrum_kind: SpectrumKind,
    radicand: int,
    repeated: bool,
) -> dict[Branch, arb] | None:
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
    lower_candidate = (trace_ball - root) / 2
    lower_ball: arb | None = lower_candidate
    if spectrum_kind == "SINGULAR_VALUES":
        if not upper > 0:
            return None
        upper = upper.sqrt()
        lower_ball = lower_candidate.sqrt() if lower_candidate > 0 else None
    result: dict[Branch, arb] = {"UPPER": upper}
    if lower_ball is not None:
        result["LOWER"] = lower_ball
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
        with flint_workprec(precision):
            balls = _branch_balls(
                trace,
                determinant,
                spectrum_kind,
                radicand,
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
    require_square_free_radicand(matrix.radicand, location=("matrix", "radicand"))
    rational, radical, trace, determinant = _polynomial_parts(matrix, spectrum_kind)
    radicand = matrix.radicand
    discriminant = _subtract(
        _multiply(trace, trace, radicand),
        _scale(determinant, Fraction(4)),
    )
    discriminant_sign = _sign(discriminant, radicand)
    if discriminant_sign < 0:  # pragma: no cover
        raise RuntimeError("a real symmetric 2 by 2 spectrum had negative discriminant")

    polynomial = _sympy_polynomial(annihilating_coefficients(matrix, spectrum_kind))
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

    return RealQuadraticSpectrum._from_kernel(
        matrix=matrix,
        spectrum_kind="SYMMETRIC_EIGENVALUES",
        values=spectrum_rows(matrix, "SYMMETRIC_EIGENVALUES"),
    )


def singular_spectrum(matrix: RealQuadraticMatrix) -> RealQuadraticSpectrum:
    """Return the exact descending singular-value spectrum of a 2 by 2 matrix."""

    return RealQuadraticSpectrum._from_kernel(
        matrix=matrix,
        spectrum_kind="SINGULAR_VALUES",
        values=spectrum_rows(matrix, "SINGULAR_VALUES"),
    )


def _swap_symmetric[PairT: QuadraticEntry](
    matrix: list[list[PairT]], left: int, right: int
) -> None:
    if left == right:
        return
    matrix[left], matrix[right] = matrix[right], matrix[left]
    for row in matrix:
        row[left], row[right] = row[right], row[left]


def _find_off_diagonal[PairT: QuadraticEntry](
    matrix: list[list[PairT]], index: int
) -> tuple[int, int] | None:
    for row in range(index, len(matrix)):
        for column in range(row + 1, len(matrix)):
            if not _is_zero(matrix[row][column]):
                return row, column
    return None


def _pair_add(left: IntegralQuadratic, right: IntegralQuadratic) -> IntegralQuadratic:
    return left[0] + right[0], left[1] + right[1]


def _pair_neg(value: IntegralQuadratic) -> IntegralQuadratic:
    return -value[0], -value[1]


def _pair_sub(left: IntegralQuadratic, right: IntegralQuadratic) -> IntegralQuadratic:
    return left[0] - right[0], left[1] - right[1]


def _pair_mul(
    left: IntegralQuadratic, right: IntegralQuadratic, radicand: int
) -> IntegralQuadratic:
    return (
        left[0] * right[0] + radicand * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _pair_exact_div(
    numerator: IntegralQuadratic, denominator: IntegralQuadratic, radicand: int
) -> IntegralQuadratic:
    """Divide in ``ZZ[sqrt(d)]``, which is exact for Bareiss elimination steps.

    Sylvester's determinant identity keeps every such quotient integral over
    any integral domain; a remainder here would refute the elimination
    invariant rather than a caller input.
    """

    num_real, num_imag = numerator
    den_real, den_imag = denominator
    norm = den_real * den_real - radicand * den_imag * den_imag
    if norm == 0:
        raise RuntimeError("fraction-free inertia elimination divided by zero")
    real = num_real * den_real - radicand * num_imag * den_imag
    imag = num_imag * den_real - num_real * den_imag
    quotient_real, real_reminder = divmod(real, norm)
    quotient_imag, imag_reminder = divmod(imag, norm)
    if real_reminder or imag_reminder:
        raise RuntimeError("fraction-free inertia elimination met an inexact division")
    return quotient_real, quotient_imag


def _clear_quadratic_denominators(
    entries: list[list[Quadratic]],
) -> list[list[IntegralQuadratic]]:
    """Scale by a positive diagonal congruence to integral pairs.

    With ``D[i]`` a common multiple of the denominators in row and column
    ``i``, ``diag(D) * M * diag(D)`` is integral and, by Sylvester's law of
    inertia, has the same inertia as ``M``. Zero patterns are preserved, so
    pivot selection is unchanged.
    """

    order = len(entries)
    scales: list[int] = []
    for index in range(order):
        scale = 1
        for other in range(order):
            for part in (
                entries[index][other][0],
                entries[index][other][1],
                entries[other][index][0],
                entries[other][index][1],
            ):
                scale = lcm(scale, part.denominator)
        scales.append(scale)
    cleared: list[list[IntegralQuadratic]] = []
    for row in range(order):
        cleared_row: list[IntegralQuadratic] = []
        for column in range(order):
            factor = scales[row] * scales[column]
            rational, radical = entries[row][column]
            cleared_row.append(
                (
                    rational.numerator * (factor // rational.denominator),
                    radical.numerator * (factor // radical.denominator),
                )
            )
        cleared.append(cleared_row)
    return cleared


def _bareiss_eliminate_one(
    matrix: list[list[IntegralQuadratic]],
    index: int,
    pivot: int,
    previous: IntegralQuadratic,
    radicand: int,
) -> tuple[int, IntegralQuadratic]:
    """Eliminate below a 1x1 Bareiss pivot, returning its congruence sign."""

    _swap_symmetric(matrix, index, pivot)
    diagonal = matrix[index][index]
    # The Bareiss diagonal holds the minor ratio whose sign, divided by the
    # previous pivot's sign, is the congruence-diagonal sign: for -I the raw
    # diagonal reads (+1) while the true second pivot is (-1).
    sign = _sign((Fraction(diagonal[0]), Fraction(diagonal[1])), radicand) * _sign(
        (Fraction(previous[0]), Fraction(previous[1])), radicand
    )
    # Unlike field elimination, rows with a zero multiplier must still be
    # scaled by pivot/previous: the Bareiss update is (m*p - l*t)/prev, not
    # m - (l/p)*t, so skipping them would miss the exact p/prev scaling (and
    # its sign when that ratio is negative).
    for row in range(index + 1, len(matrix)):
        for column in range(index + 1, len(matrix)):
            matrix[row][column] = _pair_exact_div(
                _pair_sub(
                    _pair_mul(matrix[row][column], diagonal, radicand),
                    _pair_mul(matrix[row][index], matrix[index][column], radicand),
                ),
                previous,
                radicand,
            )
    # Zero the eliminated cross only after every row has read the pivot row.
    for row in range(index + 1, len(matrix)):
        matrix[row][index] = (0, 0)
        matrix[index][row] = (0, 0)
    return sign, diagonal


def _bareiss_eliminate_two(
    matrix: list[list[IntegralQuadratic]],
    index: int,
    previous: IntegralQuadratic,
    radicand: int,
) -> IntegralQuadratic:
    """Eliminate below a 2x2 Bareiss pivot block; returns its determinant.

    The pivot search guarantees every remaining diagonal is zero (the
    maintained sibling kernel records the same invariant), so the block is
    ``[[0, b], [b, 0]]`` with determinant ``-b*b`` and inertia ``(1, 1, 0)``.
    """

    off_diagonal = _find_off_diagonal(matrix, index)
    if off_diagonal is None:  # pragma: no cover
        raise RuntimeError("2 by 2 pivot requested without an off-diagonal entry")
    first, second = off_diagonal
    _swap_symmetric(matrix, index, first)
    if second == index:
        second = first
    _swap_symmetric(matrix, index + 1, second)
    if matrix[index][index] != (0, 0) or matrix[index + 1][index + 1] != (0, 0):
        raise RuntimeError("fraction-free 2 by 2 pivot met a nonzero diagonal")
    pivot = matrix[index][index + 1]
    determinant = _pair_neg(_pair_mul(pivot, pivot, radicand))
    adjoint_00 = (0, 0)
    adjoint_01 = _pair_neg(pivot)
    adjoint_11 = (0, 0)
    if index + 2 < len(matrix):
        for row in range(index + 2, len(matrix)):
            left = matrix[row][index]
            right = matrix[row][index + 1]
            coefficient_0 = _pair_add(
                _pair_mul(left, adjoint_00, radicand),
                _pair_mul(right, adjoint_01, radicand),
            )
            coefficient_1 = _pair_add(
                _pair_mul(left, adjoint_01, radicand),
                _pair_mul(right, adjoint_11, radicand),
            )
            for column in range(index + 2, len(matrix)):
                top = matrix[index][column]
                bottom = matrix[index + 1][column]
                numerator = _pair_sub(
                    _pair_mul(matrix[row][column], determinant, radicand),
                    _pair_add(
                        _pair_mul(coefficient_0, top, radicand),
                        _pair_mul(coefficient_1, bottom, radicand),
                    ),
                )
                matrix[row][column] = _pair_exact_div(
                    _pair_exact_div(numerator, previous, radicand),
                    previous,
                    radicand,
                )
        # Zero the eliminated cross only after every row has read the pivots.
        for row in range(index + 2, len(matrix)):
            matrix[row][index] = (0, 0)
            matrix[row][index + 1] = (0, 0)
            matrix[index][row] = (0, 0)
            matrix[index + 1][row] = (0, 0)
    # The block determinant is itself still scaled by the prior pivot.  Carry
    # the normalized determinant as the next Bareiss divisor so that a later
    # block receives the same minor invariant as a sequence of 1x1 pivots.
    return _pair_exact_div(determinant, previous, radicand)


def _bareiss_inertia_counts(
    pairs: list[list[IntegralQuadratic]], radicand: int
) -> tuple[int, int, int]:
    """Count inertia by fraction-free symmetric elimination over ``ZZ[√d]``.

    Entries stay integral: every update is a ratio of minors, hence exactly
    divisible by the previous pivot (Sylvester's identity over an integral
    domain). Hadamard's bound controls intermediate growth, replacing the
    per-operation Fraction normalization of field elimination.
    """

    matrix = [row[:] for row in pairs]
    positive = negative = zero = 0
    index = 0
    previous: IntegralQuadratic = (1, 0)
    while index < len(matrix):
        pivot = next(
            (row for row in range(index, len(matrix)) if matrix[row][row] != (0, 0)),
            None,
        )
        if pivot is not None:
            sign, previous = _bareiss_eliminate_one(
                matrix, index, pivot, previous, radicand
            )
            positive += sign > 0
            negative += sign < 0
            index += 1
            continue
        if _find_off_diagonal(matrix, index) is None:
            zero += len(matrix) - index
            break
        previous = _bareiss_eliminate_two(matrix, index, previous, radicand)
        positive += 1
        negative += 1
        index += 2
    return positive, negative, zero


def _inertia_counts(matrix: RealQuadraticMatrix) -> tuple[int, int, int]:
    radicand = matrix.radicand
    reduced = _matrix_entries(matrix)
    if all(
        reduced[row][column] == _ZERO
        for row in range(len(reduced))
        for column in range(row + 1, len(reduced))
    ):
        signs = [
            _sign(reduced[index][index], radicand) for index in range(len(reduced))
        ]
        return (
            sum(sign > 0 for sign in signs),
            sum(sign < 0 for sign in signs),
            sum(sign == 0 for sign in signs),
        )
    return _bareiss_inertia_counts(_clear_quadratic_denominators(reduced), radicand)


def inertia_data(
    matrix: RealQuadraticMatrix,
) -> tuple[int, int, int, Definiteness]:
    """Return exact Sylvester inertia data for an admitted matrix."""

    require_inertia_matrix(matrix)
    require_square_free_radicand(matrix.radicand, location=("matrix", "radicand"))
    positive, negative, zero = _inertia_counts(matrix)
    return (
        positive,
        negative,
        zero,
        _definiteness_from_inertia(positive, negative, zero),
    )


def inertia(matrix: RealQuadraticMatrix) -> RealQuadraticInertia:
    """Return exact Sylvester inertia for a symmetric real-quadratic matrix."""

    positive, negative, zero, definiteness = inertia_data(matrix)
    return RealQuadraticInertia._from_kernel(
        matrix=matrix,
        n_positive=positive,
        n_negative=negative,
        n_zero=zero,
        definiteness=definiteness,
    )


def verify_symmetric_spectrum(claim: RealQuadraticSpectrum) -> bool:
    """Verify a serialized exact symmetric spectrum against its source."""
    if not isinstance(claim, RealQuadraticSpectrum):
        return False
    if claim.spectrum_kind != "SYMMETRIC_EIGENVALUES":
        return False
    try:
        return symmetric_spectrum(claim.matrix) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_singular_spectrum(claim: RealQuadraticSpectrum) -> bool:
    """Verify a serialized exact singular spectrum against its source."""
    if not isinstance(claim, RealQuadraticSpectrum):
        return False
    if claim.spectrum_kind != "SINGULAR_VALUES":
        return False
    try:
        return singular_spectrum(claim.matrix) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_inertia(claim: RealQuadraticInertia) -> bool:
    """Verify serialized inertia counts and definiteness against the source."""
    if not isinstance(claim, RealQuadraticInertia):
        return False
    try:
        return inertia(claim.matrix) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


__all__ = [
    "inertia",
    "singular_spectrum",
    "symmetric_spectrum",
    "verify_inertia",
    "verify_singular_spectrum",
    "verify_symmetric_spectrum",
]
