"""Native exact shift Ore-operator arithmetic over QQ(n)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial, gcd, lcm
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
    MAX_SEQUENCE_TOTAL_DIGITS,
)
from jacobian.math.ore_algebras._models import (
    MAX_COEFFICIENT_RECURRENCE_OUTPUT_BYTES,
    MAX_COEFFICIENT_RECURRENCE_WORK_CELLS,
    MAX_DFINITE_PREFIX_OUTPUT_BYTES,
    MAX_DFINITE_PREFIX_SCALAR_BITS,
    MAX_DFINITE_PREFIX_WORK_UNITS,
    MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_BYTES,
    MAX_DIFFERENTIAL_ADDITIVE_WORK_CELLS,
    MAX_DIFFERENTIAL_ORDER,
    MAX_DIFFERENTIAL_TERMS,
    MAX_RECURRENCE_PREFIX_OUTPUT_BYTES,
    MAX_RECURRENCE_PREFIX_WORK_CELLS,
    MAX_SHIFT_ADDITIVE_OUTPUT_BYTES,
    MAX_SHIFT_ADDITIVE_WORK_CELLS,
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    MAX_SHIFT_LEDGER_ROWS,
    MAX_SHIFT_ORDER,
    MAX_SHIFT_POWER_EXPONENT,
    MAX_SHIFT_POWER_RESULT_ORDER,
    MAX_SHIFT_POWER_WORK_CELLS,
    MAX_SHIFT_PREFIX_EVALUATION_CELLS,
    MAX_SHIFT_PREFIX_INDEX,
    MAX_SHIFT_PREFIX_OUTPUT_BYTES,
    MAX_SHIFT_PREFIX_WORK_UNITS,
    MAX_SHIFT_RESULT_DEGREE,
    MAX_SHIFT_RESULT_DIGITS,
    MAX_SHIFT_RESULT_ORDER,
    MAX_SHIFT_TERMS,
    CoefficientRecurrenceBoundaryRow,
    CoefficientRecurrenceBoundaryTerm,
    DFinitePowerSeries,
    DFinitePowerSeriesPrefixRequest,
    DFinitePowerSeriesRequest,
    DifferentialCoefficientRecurrence,
    DifferentialOperatorAddResult,
    DifferentialOperatorApplyResult,
    DifferentialOperatorMultiplyResult,
    DifferentialOperatorNormalizeResult,
    DifferentialOreOperator,
    PolynomialRecurrencePrefix,
    PolynomialRecurrencePrefixRequest,
    ShiftMultiplyLedgerRow,
    ShiftOperatorAddResult,
    ShiftOperatorMultiplyResult,
    ShiftOperatorNormalizeResult,
    ShiftOperatorPowerResult,
    ShiftOperatorPrefixContribution,
    ShiftOperatorPrefixPoleExclusion,
    ShiftOperatorPrefixResidual,
    ShiftOperatorPrefixResult,
    ShiftOperatorScalarMultiplyResult,
    ShiftOreOperator,
    ShiftOreTerm,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_canonical_rational_function,
)

_Poly = dict[int, Fraction]


def _falling_factorial_polynomial(offset: int, order: int) -> _Poly:
    """Expand ``(n + offset)_falling_order`` in QQ[n]."""
    result: _Poly = {0: Fraction(1)}
    for factor_index in range(order):
        factor = {
            0: Fraction(offset - factor_index),
            1: Fraction(1),
        }
        result = _poly_mul(result, factor)
    return result


@dataclass(frozen=True)
class _ShiftProductCell:
    left_exponent: int
    right_exponent: int
    shifted: tuple[_Poly, _Poly]
    contribution: tuple[_Poly, _Poly]


def _as_operator(value: ShiftOreOperator | Mapping[str, Any]) -> ShiftOreOperator:
    try:
        return (
            value
            if isinstance(value, ShiftOreOperator)
            else ShiftOreOperator.model_validate(value)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.shift_operator",
            message="the shift operator must be a valid typed value over QQ(n)",
        ) from exc


def _run_admission(admission: Any, *, location: tuple[str | int, ...]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except OperationResourceAdmissionError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="ore_algebra.admission",
            message=str(exc),
        ) from exc


def _decode_poly(terms: tuple[RationalPolynomialTerm, ...]) -> _Poly:
    return {int(term.exponents[0]): term.coefficient.as_fraction() for term in terms}


def _decode_rf(value: RationalFunction) -> tuple[_Poly, _Poly]:
    return (
        _decode_poly(value.numerator.terms),
        _decode_poly(value.denominator.terms),
    )


def _require_rf_carrier_height(
    value: tuple[_Poly, _Poly], *, location: tuple[str | int, ...]
) -> None:
    """Reject a result before constructing a RationalFunction carrier."""

    for polynomial in value:
        if any(
            max(len(str(abs(coefficient.numerator))), len(str(coefficient.denominator)))
            > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            for coefficient in polynomial.values()
        ):
            raise OperationResourceAdmissionError(
                location=location,
                code="ore_algebra.shift_product_coefficient_digits",
                message="shift-operator product exceeds the rational-function coefficient-digit carrier",
            )


def _admit_shift_operator(
    operator: ShiftOreOperator, *, label: str
) -> ShiftOreOperator:
    try:
        value = ShiftOreOperator.model_validate(operator.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="ore_algebra.shift_operator",
            message="the shift operator must be canonical over QQ(n)",
        ) from exc
    for index, term in enumerate(value.terms):
        _run_admission(
            lambda term=term: require_canonical_rational_function(
                term.coefficient,
                maximum_terms=MAX_SHIFT_COEFFICIENT_TERMS,
                maximum_exponent=MAX_SHIFT_COEFFICIENT_DEGREE,
                maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
                label=f"{label} coefficient",
            ),
            location=(label, "terms", index),
        )
    return value


def _admit_shift_multiply(
    left: ShiftOreOperator, right: ShiftOreOperator
) -> tuple[ShiftOreOperator, ShiftOreOperator]:
    left = _admit_shift_operator(left, label="left")
    right = _admit_shift_operator(right, label="right")
    rows = len(left.terms) * len(right.terms)
    if rows > MAX_SHIFT_LEDGER_ROWS:
        raise OperationResourceAdmissionError(
            location=("left",),
            code="ore_algebra.shift_product_pairs",
            message="shift-operator product exceeds the term-pair budget",
        )
    if (
        left.order >= 0
        and right.order >= 0
        and left.order + right.order > MAX_SHIFT_RESULT_ORDER
    ):
        raise OperationResourceAdmissionError(
            location=("left",),
            code="ore_algebra.shift_product_order",
            message="shift-operator product exceeds the result-order budget",
        )
    _admit_shift_product_degree_bounds(left, right)
    return left, right


def _admit_shift_product_degree_bounds(
    left: ShiftOreOperator, right: ShiftOreOperator
) -> None:
    """Bound every RF stage before any ledger cell is constructed.

    A shifted coefficient preserves numerator and denominator degree.  A pair
    contribution multiplies those parts, while the sum of contributions at
    one shift exponent can use the product of all pair denominators as a
    common denominator.  These are deliberately upper bounds: normalization
    and a GCD can only lower the degrees, so no backend work is needed for
    admission and a later exact cancellation cannot make an over-budget
    request admissible by surprise.
    """

    grouped: dict[int, list[tuple[int, int]]] = {}
    for left_term in left.terms:
        left_numerator, left_denominator = _decode_rf(left_term.coefficient)
        left_numerator_degree = _poly_degree(left_numerator)
        left_denominator_degree = _poly_degree(left_denominator)
        for right_term in right.terms:
            right_numerator, right_denominator = _decode_rf(right_term.coefficient)
            right_numerator_degree = _poly_degree(right_numerator)
            right_denominator_degree = _poly_degree(right_denominator)

            # sigma^i preserves degree, including when its dense expansion is
            # later normalized.  Keep this check separate so a future change
            # to the input envelope cannot bypass the shifted-cell bound.
            if (
                max(right_numerator_degree, right_denominator_degree)
                > MAX_SHIFT_RESULT_DEGREE
            ):
                raise OperationResourceAdmissionError(
                    location=("right", "terms", right_term.exponent),
                    code="ore_algebra.shift_coefficient_degree",
                    message="shifted coefficient exceeds the result-degree budget",
                )

            contribution_numerator_degree = (
                left_numerator_degree + right_numerator_degree
            )
            contribution_denominator_degree = (
                left_denominator_degree + right_denominator_degree
            )
            if (
                max(contribution_numerator_degree, contribution_denominator_degree)
                > MAX_SHIFT_RESULT_DEGREE
            ):
                raise OperationResourceAdmissionError(
                    location=("left", "terms", left_term.exponent),
                    code="ore_algebra.shift_contribution_degree",
                    message="shift-operator pair contribution exceeds the result-degree budget",
                )

            result_exponent = left_term.exponent + right_term.exponent
            # A shift by i expands p(n+i); for degree d and coefficient
            # height h, each binomial contribution is bounded by h+i*d bits.
            # Multiplication by the left coefficient adds its height. Preflight
            # before _plan_shift_product_cells materializes the shifted product.
            left_numerator_height = max(
                (abs(v.numerator).bit_length() for v in left_numerator.values()),
                default=0,
            )
            left_denominator_height = max(
                (abs(v.numerator).bit_length() for v in left_denominator.values()),
                default=0,
            )
            for degree, right_height, left_height in (
                (
                    right_numerator_degree,
                    max(
                        (
                            abs(v.numerator).bit_length()
                            for v in _decode_rf(right_term.coefficient)[0].values()
                        ),
                        default=0,
                    ),
                    left_numerator_height,
                ),
                (
                    right_denominator_degree,
                    max(
                        (
                            abs(v.numerator).bit_length()
                            for v in _decode_rf(right_term.coefficient)[1].values()
                        ),
                        default=0,
                    ),
                    left_denominator_height,
                ),
            ):
                if (
                    right_height + left_term.exponent * degree + left_height
                    > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS * 4
                ):
                    raise OperationResourceAdmissionError(
                        location=("right", "terms", right_term.exponent),
                        code="ore_algebra.shift_product_coefficient_digits",
                        message="shifted coefficient exceeds the rational-function coefficient-digit carrier",
                    )
            grouped.setdefault(result_exponent, []).append(
                (contribution_numerator_degree, contribution_denominator_degree)
            )

    for result_exponent, contributions in grouped.items():
        common_denominator_degree = sum(
            denominator_degree for _, denominator_degree in contributions
        )
        accumulated_numerator_degree = max(
            numerator_degree + common_denominator_degree - denominator_degree
            for numerator_degree, denominator_degree in contributions
        )
        # A common denominator and its lifted numerator are the pre-
        # normalization/GCD representation of the accumulated RF.  They are
        # also the maximum degree any ledger replay can expose.
        if (
            max(common_denominator_degree, accumulated_numerator_degree)
            > MAX_SHIFT_RESULT_DEGREE
        ):
            raise OperationResourceAdmissionError(
                location=("product", result_exponent),
                code="ore_algebra.shift_product_degree",
                message="shift-operator product exceeds the coefficient-degree budget",
            )
        if (
            max(common_denominator_degree, accumulated_numerator_degree) + 1
            > MAX_RATIONAL_FUNCTION_TERMS
        ):
            raise OperationResourceAdmissionError(
                location=("product", result_exponent),
                code="ore_algebra.shift_product_terms",
                message="shift-operator product exceeds the rational-function term budget",
            )


def _poly_degree(poly: _Poly) -> int:
    return max(poly, default=-1)


def _shift_poly(poly: _Poly, step: int) -> _Poly:
    """Compute the exact shifted polynomial p(n + step)."""
    if step == 0 or not poly:
        return dict(poly)
    shifted: _Poly = {}
    for exponent, coefficient in poly.items():
        for target in range(exponent + 1):
            shifted[target] = shifted.get(target, Fraction(0)) + coefficient * comb(
                exponent, target
            ) * (step ** (exponent - target))
    return {exponent: value for exponent, value in shifted.items() if value != 0}


def _poly_add(left: _Poly, right: _Poly) -> _Poly:
    combined = dict(left)
    for exponent, value in right.items():
        total = combined.get(exponent, Fraction(0)) + value
        if total == 0:
            combined.pop(exponent, None)
        else:
            combined[exponent] = total
    return combined


def _poly_mul(left: _Poly, right: _Poly) -> _Poly:
    product: _Poly = {}
    for left_exponent, left_value in left.items():
        for right_exponent, right_value in right.items():
            exponent = left_exponent + right_exponent
            product[exponent] = (
                product.get(exponent, Fraction(0)) + left_value * right_value
            )
    return {exponent: value for exponent, value in product.items() if value != 0}


def _poly_dense(poly: _Poly) -> list[Fraction]:
    if not poly:
        return []
    degree = max(poly)
    return [poly.get(exponent, Fraction(0)) for exponent in range(degree + 1)]


def _poly_divmod(
    dividend: list[Fraction], divisor: list[Fraction]
) -> tuple[list[Fraction], list[Fraction]]:
    remainder = list(dividend)
    quotient: list[Fraction] = [Fraction(0)] * max(len(dividend) - len(divisor) + 1, 0)
    while len(remainder) >= len(divisor) and any(remainder):
        while remainder and remainder[-1] == 0:
            remainder.pop()
        if len(remainder) < len(divisor):
            break
        factor = remainder[-1] / divisor[-1]
        shift = len(remainder) - len(divisor)
        quotient[shift] = factor
        for index, value in enumerate(divisor):
            remainder[shift + index] -= factor * value
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return quotient, remainder


def _poly_gcd(left: _Poly, right: _Poly) -> _Poly:
    current = _poly_dense(left)
    other = _poly_dense(right)
    while any(other):
        _, remainder = _poly_divmod(current, other)
        current, other = other, remainder
    while current and current[-1] == 0:
        current.pop()
    if not current:
        return {}
    leading = current[-1]
    return {index: value / leading for index, value in enumerate(current) if value != 0}


def _poly_exact_div(poly: _Poly, divisor: _Poly) -> _Poly:
    quotient, remainder = _poly_divmod(_poly_dense(poly), _poly_dense(divisor))
    if any(remainder):
        raise ValueError("polynomial division left a nonzero remainder")
    while quotient and quotient[-1] == 0:
        quotient.pop()
    return {index: value for index, value in enumerate(quotient) if value != 0}


def _normalize(numerator: _Poly, denominator: _Poly) -> tuple[_Poly, _Poly]:
    if not denominator:
        raise ValueError("rational-function denominator cannot be zero")
    if not numerator:
        return {}, {0: Fraction(1)}
    common = _poly_gcd(numerator, denominator)
    if common and max(common) > 0:
        numerator = _poly_exact_div(numerator, common)
        denominator = _poly_exact_div(denominator, common)
    leading = denominator[max(denominator)]
    return (
        {exponent: value / leading for exponent, value in numerator.items()},
        {exponent: value / leading for exponent, value in denominator.items()},
    )


def _rf_add(
    left: tuple[_Poly, _Poly], right: tuple[_Poly, _Poly]
) -> tuple[_Poly, _Poly]:
    (left_num, left_den), (right_num, right_den) = left, right
    return _normalize(
        _poly_add(_poly_mul(left_num, right_den), _poly_mul(right_num, left_den)),
        _poly_mul(left_den, right_den),
    )


def _rf_mul(
    left: tuple[_Poly, _Poly], right: tuple[_Poly, _Poly]
) -> tuple[_Poly, _Poly]:
    (left_num, left_den), (right_num, right_den) = left, right
    return _normalize(_poly_mul(left_num, right_num), _poly_mul(left_den, right_den))


def _rf_shift(value: tuple[_Poly, _Poly], step: int) -> tuple[_Poly, _Poly]:
    numerator, denominator = value
    return _normalize(_shift_poly(numerator, step), _shift_poly(denominator, step))


def _encode_poly(poly: _Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(value),
                exponents=(exponent,),
            )
            for exponent, value in sorted(poly.items(), reverse=True)
        )
    )


def _encode_rf(value: tuple[_Poly, _Poly], variable: str = "n") -> RationalFunction:
    numerator, denominator = value
    return RationalFunction(
        domain="QQ",
        variables=(variable,),
        numerator=_encode_poly(numerator),
        denominator=_encode_poly(denominator),
    )


def _encode_differential_rf(value: tuple[_Poly, _Poly]) -> RationalFunction:
    """Encode an admitted differential result without leaking carrier errors."""
    try:
        return _encode_rf(value, "x")
    except Exception as exc:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.differential_result_carrier",
            message="differential Ore output exceeds the rational-function carrier budget",
        ) from exc


def _plan_shift_product_cells(
    left: ShiftOreOperator, right: ShiftOreOperator
) -> tuple[_ShiftProductCell, ...]:
    """Expand each admitted pair exactly once into a request-local plan."""

    cells: list[_ShiftProductCell] = []
    for left_term in left.terms:
        left_coefficient = _decode_rf(left_term.coefficient)
        for right_term in right.terms:
            shifted = _rf_shift(_decode_rf(right_term.coefficient), left_term.exponent)
            _require_rf_carrier_height(
                shifted,
                location=("right", "terms", right_term.exponent, "shifted_coefficient"),
            )
            contribution = _rf_mul(left_coefficient, shifted)
            _require_rf_carrier_height(
                contribution,
                location=("left", "terms", left_term.exponent, "contribution"),
            )
            cells.append(
                _ShiftProductCell(
                    left_exponent=left_term.exponent,
                    right_exponent=right_term.exponent,
                    shifted=shifted,
                    contribution=contribution,
                )
            )
    return tuple(cells)


def shift_operator_multiply(
    left: ShiftOreOperator | Mapping[str, Any],
    right: ShiftOreOperator | Mapping[str, Any],
) -> ShiftOperatorMultiplyResult:
    """Multiply two shift operators via S^i a(n) = a(n+i) S^i."""
    left_value = _as_operator(left)
    right_value = _as_operator(right)
    left_value, right_value = _admit_shift_multiply(left_value, right_value)
    cells = _plan_shift_product_cells(left_value, right_value)
    accumulated: dict[int, tuple[_Poly, _Poly]] = {}
    for cell in cells:
        result_exponent = cell.left_exponent + cell.right_exponent
        if result_exponent in accumulated:
            accumulated[result_exponent] = _rf_add(
                accumulated[result_exponent], cell.contribution
            )
        else:
            accumulated[result_exponent] = cell.contribution
        _require_rf_carrier_height(
            accumulated[result_exponent],
            location=("product", result_exponent),
        )
    ledger = tuple(
        ShiftMultiplyLedgerRow.model_construct(
            left_exponent=cell.left_exponent,
            right_exponent=cell.right_exponent,
            result_exponent=cell.left_exponent + cell.right_exponent,
            shifted_coefficient=_encode_rf(cell.shifted),
            contribution=_encode_rf(cell.contribution),
        )
        for cell in cells
    )
    product_terms = []
    for exponent in sorted(accumulated):
        numerator, denominator = accumulated[exponent]
        if not numerator:
            continue
        _require_rf_carrier_height(
            (numerator, denominator), location=("product", exponent)
        )
        for value in (*numerator.values(), *denominator.values()):
            digits = max(len(str(abs(value.numerator))), len(str(value.denominator)))
            if digits > MAX_SHIFT_RESULT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("product",),
                    code="ore_algebra.shift_product_coefficient_digits",
                    message="shift-operator product exceeds the result-digit budget",
                )
        product_terms.append(
            {"exponent": exponent, "coefficient": _encode_rf(accumulated[exponent])}
        )
    product = ShiftOreOperator.model_validate({"variable": "n", "terms": product_terms})
    return ShiftOperatorMultiplyResult._from_kernel(
        left_value,
        right_value,
        product=product,
        ledger=tuple(ledger),
    )


def _shift_power_source_profile(
    operator: ShiftOreOperator,
) -> dict[int, tuple[int, int, int]]:
    """Read degree, support, and height bounds for integer-polynomial terms."""

    source: dict[int, tuple[int, int, int]] = {}
    for term in operator.terms:
        if _decode_poly(term.coefficient.denominator.terms) != {0: Fraction(1)}:
            raise OperationResourceAdmissionError(
                location=("operator", "terms", term.exponent, "coefficient"),
                code="ore_algebra.shift_power_coefficient_envelope",
                message=(
                    "powers above one currently require integer-polynomial "
                    "coefficients in ZZ[n] so every stage's coefficient "
                    "growth can be bounded before expansion"
                ),
            )
        coefficients = tuple(
            coefficient.coefficient.as_fraction()
            for coefficient in term.coefficient.numerator.terms
        )
        if any(coefficient.denominator != 1 for coefficient in coefficients):
            raise OperationResourceAdmissionError(
                location=("operator", "terms", term.exponent, "coefficient"),
                code="ore_algebra.shift_power_coefficient_envelope",
                message=(
                    "powers above one currently require integer-polynomial "
                    "coefficients in ZZ[n] so every stage's coefficient "
                    "growth can be bounded before expansion"
                ),
            )
        source[term.exponent] = (
            max(
                (
                    max(poly_term.exponents)
                    for poly_term in term.coefficient.numerator.terms
                ),
                default=0,
            ),
            len(term.coefficient.numerator.terms),
            max(
                (
                    len(format_canonical_integer(abs(coefficient.numerator)))
                    for coefficient in coefficients
                ),
                default=0,
            ),
        )
    return source


def _preflight_shift_power_stages(operator: ShiftOreOperator, exponent: int) -> None:
    """Bound every degree, support, height, and product stage before work."""

    source = _shift_power_source_profile(operator)
    stages = dict(source)
    planned_work = 0
    for stage in range(1, exponent):
        next_stage: dict[int, tuple[int, int, int, int]] = {}
        planned_work += len(stages) * len(source)
        for left_exponent, (left_degree, left_terms, left_digits) in stages.items():
            for right_exponent, (
                right_degree,
                right_terms,
                right_digits,
            ) in source.items():
                pair = _shift_power_pair_profile(
                    left_exponent,
                    left_degree,
                    left_terms,
                    left_digits,
                    right_degree,
                    right_terms,
                    right_digits,
                )
                output_exponent = left_exponent + right_exponent
                current = next_stage.get(output_exponent)
                if current is None:
                    next_stage[output_exponent] = (*pair, 1)
                else:
                    degree, terms, digits, pair_count = current
                    next_stage[output_exponent] = (
                        max(degree, pair[0]),
                        terms + pair[1],
                        max(digits, pair[2]),
                        pair_count + 1,
                    )
        if planned_work > MAX_SHIFT_POWER_WORK_CELLS:
            raise OperationResourceAdmissionError(
                location=("exponent", stage + 1),
                code="ore_algebra.shift_power_work",
                message="shift-operator power exceeds its admitted product-cell work bound",
            )
        if len(next_stage) > MAX_SHIFT_TERMS:
            raise OperationResourceAdmissionError(
                location=("exponent", stage + 1),
                code="ore_algebra.shift_power_term_bound",
                message="a shift-operator power stage may exceed the sparse term bound",
            )
        stages = _finish_shift_power_stage(next_stage, stage + 1)


def _shift_power_pair_profile(
    left_exponent: int,
    left_degree: int,
    left_terms: int,
    left_digits: int,
    right_degree: int,
    right_terms: int,
    right_digits: int,
) -> tuple[int, int, int]:
    """Bound one shift-weighted polynomial product before expansion."""

    shift_digits = 0
    if left_exponent and right_degree:
        shift_factor = (right_degree + 1) * (2 * left_exponent) ** right_degree
        shift_digits = len(str(shift_factor - 1))
    shifted_terms = (
        right_terms if not left_exponent or right_degree == 0 else right_degree + 1
    )
    convolution_count = min(left_degree + 1, right_degree + 1)
    pair_terms = min(
        left_degree + right_degree + 1,
        left_terms * shifted_terms,
    )
    pair_digits = (
        left_digits
        + right_digits
        + shift_digits
        + (len(str(convolution_count - 1)) if convolution_count > 1 else 0)
    )
    return left_degree + right_degree, pair_terms, pair_digits


def _finish_shift_power_stage(
    stage: dict[int, tuple[int, int, int, int]], stage_number: int
) -> dict[int, tuple[int, int, int]]:
    """Check one predicted stage against reusable operator input bounds."""

    result: dict[int, tuple[int, int, int]] = {}
    for exponent, (degree, terms, digits, pair_count) in stage.items():
        digits += len(str(pair_count - 1)) if pair_count > 1 else 0
        terms = min(degree + 1, terms)
        if (
            degree > MAX_SHIFT_COEFFICIENT_DEGREE
            or terms > MAX_SHIFT_COEFFICIENT_TERMS
            or digits > MAX_SHIFT_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("exponent", stage_number),
                code="ore_algebra.shift_power_intermediate_bound",
                message=(
                    "a shift-operator power stage may exceed the coefficient "
                    "degree, term, or digit bound before completion"
                ),
            )
        result[exponent] = (degree, terms, digits)
    return result


def shift_operator_power(
    operator: ShiftOreOperator | Mapping[str, Any], exponent: int
) -> ShiftOperatorPowerResult:
    """Compute a bounded nonnegative power in the shift Ore algebra."""

    value = _admit_shift_operator(_as_operator(operator), label="operator")
    if type(exponent) is not int or not 0 <= exponent <= MAX_SHIFT_POWER_EXPONENT:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="ore_algebra.shift_power_exponent",
            message=(
                "the shift-operator exponent must be in the admitted range "
                f"0..{MAX_SHIFT_POWER_EXPONENT}"
            ),
        )
    if value.order >= 0 and value.order * exponent > MAX_SHIFT_POWER_RESULT_ORDER:
        raise OperationResourceAdmissionError(
            location=("exponent",),
            code="ore_algebra.shift_power_order",
            message="shift-operator power exceeds the canonical result-order bound",
        )
    if exponent > 1:
        _preflight_shift_power_stages(value, exponent)

    one = _encode_rf(({0: Fraction(1)}, {0: Fraction(1)}))
    identity = ShiftOreOperator(
        variable="n",
        terms=(
            # This is the multiplicative identity in QQ(n)<S>.
            ShiftOreTerm(exponent=0, coefficient=one),
        ),
    )
    result = identity if exponent == 0 else value
    for _ in range(1, exponent):
        result = shift_operator_multiply(result, value).product
        if not result.terms:
            break
    return ShiftOperatorPowerResult._from_kernel(value, exponent, result)


def _evaluate_rational_polynomial_at(
    polynomial: SparseRationalPolynomial, index: int
) -> Fraction:
    """Evaluate one admitted univariate rational polynomial exactly."""
    return sum(
        (
            term.coefficient.as_fraction() * index ** term.exponents[0]
            for term in polynomial.terms
        ),
        Fraction(0),
    )


def _digit_count(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _admit_polynomial_shift_operator(
    value: ShiftOreOperator | Mapping[str, Any], *, label: str
) -> ShiftOreOperator:
    """Admit an operator in the closed subalgebra QQ[n]<S>."""
    operator = _admit_shift_operator(_as_operator(value), label=label)
    if any(
        _decode_poly(term.coefficient.denominator.terms) != {0: Fraction(1)}
        for term in operator.terms
    ):
        raise OperationDomainValidationError(
            location=(label, "terms"),
            code="ore_algebra.shift_polynomial_coefficients",
            message="this operation requires polynomial coefficients in QQ[n]",
        )
    return operator


def _operator_representation_byte_bound(operator: ShiftOreOperator) -> int:
    return 256 * len(operator.terms) + sum(
        128
        + sum(
            _digit_count(entry.coefficient.num)
            + _digit_count(entry.coefficient.den)
            + 128
            for entry in term.coefficient.numerator.terms
        )
        for term in operator.terms
    )


def _preflight_polynomial_operator_addition(
    left: ShiftOreOperator,
    right: ShiftOreOperator,
) -> tuple[tuple[int, dict[int, Fraction] | None, dict[int, Fraction] | None], ...]:
    """Plan sparse coefficient sums and bound them before adding rationals."""
    left_coefficients = {
        term.exponent: _decode_poly(term.coefficient.numerator.terms)
        for term in left.terms
    }
    right_coefficients = {
        term.exponent: _decode_poly(term.coefficient.numerator.terms)
        for term in right.terms
    }
    exponents = tuple(sorted(left_coefficients.keys() | right_coefficients.keys()))
    if len(exponents) > MAX_SHIFT_TERMS:
        raise OperationResourceAdmissionError(
            location=("sum", "terms"),
            code="ore_algebra.shift_addition_terms",
            message="shift-operator sum exceeds the sparse operator-term budget",
        )
    cells = sum(map(len, left_coefficients.values())) + sum(
        map(len, right_coefficients.values())
    )
    if cells > MAX_SHIFT_ADDITIVE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("left", "terms"),
            code="ore_algebra.shift_addition_work",
            message="shift-operator addition exceeds the sparse coefficient-work budget",
        )

    plan = tuple(
        (exponent, left_coefficients.get(exponent), right_coefficients.get(exponent))
        for exponent in exponents
    )
    output_bytes = (
        _operator_representation_byte_bound(left)
        + _operator_representation_byte_bound(right)
        + 256 * len(plan)
    )
    for _exponent, first, second in plan:
        support = (first or {}).keys() | (second or {}).keys()
        output_bytes += 128 * len(support)
        for degree in support:
            first_value = (first or {}).get(degree)
            second_value = (second or {}).get(degree)
            if first_value is None:
                numerator_digits = _digit_count(second_value.numerator)
                denominator_digits = _digit_count(second_value.denominator)
            elif second_value is None:
                numerator_digits = _digit_count(first_value.numerator)
                denominator_digits = _digit_count(first_value.denominator)
            else:
                numerator_digits = (
                    max(
                        _digit_count(first_value.numerator)
                        + _digit_count(second_value.denominator),
                        _digit_count(second_value.numerator)
                        + _digit_count(first_value.denominator),
                    )
                    + 1
                )
                denominator_digits = _digit_count(
                    first_value.denominator
                ) + _digit_count(second_value.denominator)
            if (
                max(numerator_digits, denominator_digits)
                > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            ):
                raise OperationResourceAdmissionError(
                    location=("sum", _exponent, degree),
                    code="ore_algebra.shift_addition_coefficient_digits",
                    message="shift-operator addition can exceed the rational coefficient-digit carrier",
                )
            output_bytes += numerator_digits + denominator_digits + 64
    if output_bytes > MAX_SHIFT_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("sum",),
            code="ore_algebra.shift_addition_output",
            message="shift-operator sum exceeds its serialized output budget",
        )
    return plan


def shift_operator_add(
    left: ShiftOreOperator | Mapping[str, Any],
    right: ShiftOreOperator | Mapping[str, Any],
) -> ShiftOperatorAddResult:
    """Add polynomial-coefficient shift operators over the subring QQ[n]."""
    left_value = _admit_polynomial_shift_operator(left, label="left")
    right_value = _admit_polynomial_shift_operator(right, label="right")
    plan = _preflight_polynomial_operator_addition(left_value, right_value)
    result_terms: list[dict[str, Any]] = []
    for exponent, first, second in plan:
        coefficient = dict(first or {})
        for degree, value in (second or {}).items():
            total = coefficient.get(degree, Fraction(0)) + value
            if total:
                coefficient[degree] = total
            else:
                coefficient.pop(degree, None)
        if coefficient:
            result_terms.append(
                {
                    "exponent": exponent,
                    "coefficient": _encode_rf((coefficient, {0: Fraction(1)})),
                }
            )
    result = ShiftOreOperator.model_validate({"variable": "n", "terms": result_terms})
    return ShiftOperatorAddResult._from_kernel(left_value, right_value, result)


def shift_operator_scalar_left_multiply(
    scalar: RationalFunction | Mapping[str, Any],
    operator: ShiftOreOperator | Mapping[str, Any],
) -> ShiftOperatorScalarMultiplyResult:
    """Left-scale a polynomial-coefficient shift operator by an element of QQ."""
    operator_value = _admit_polynomial_shift_operator(operator, label="operator")
    scalar_value = _as_rational_function(scalar)
    try:
        scalar_value = RationalFunction.model_validate(scalar_value.model_dump())
        require_canonical_rational_function(
            scalar_value,
            maximum_terms=1,
            maximum_exponent=0,
            maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
            label="shift scalar",
        )
        scalar_numerator, scalar_denominator = _decode_rf(scalar_value)
        if (
            scalar_value.variables != ("n",)
            or scalar_denominator != {0: Fraction(1)}
            or any(exponent != 0 for exponent in scalar_numerator)
        ):
            raise ValueError("the scalar must be a rational constant on QQ(n)")
        scalar_fraction = scalar_numerator.get(0, Fraction(0))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("scalar",),
            code="ore_algebra.shift_rational_constant",
            message="the left scalar must be a canonical rational constant on QQ(n)",
        ) from exc

    work_cells = sum(
        len(term.coefficient.numerator.terms) for term in operator_value.terms
    )
    if work_cells > MAX_SHIFT_ADDITIVE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("operator", "terms"),
            code="ore_algebra.shift_scalar_work",
            message="shift-operator scalar multiplication exceeds its coefficient-work budget",
        )
    scalar_digits = max(
        _digit_count(scalar_fraction.numerator),
        _digit_count(scalar_fraction.denominator),
    )
    output_bytes = _operator_representation_byte_bound(operator_value) + 256
    for term in operator_value.terms:
        for coefficient in term.coefficient.numerator.terms:
            numerator_digits = _digit_count(coefficient.coefficient.num) + scalar_digits
            denominator_digits = (
                _digit_count(coefficient.coefficient.den) + scalar_digits
            )
            if (
                max(numerator_digits, denominator_digits)
                > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            ):
                raise OperationResourceAdmissionError(
                    location=("operator", "terms", term.exponent),
                    code="ore_algebra.shift_scalar_coefficient_digits",
                    message="shift-operator scalar product can exceed the rational coefficient-digit carrier",
                )
            output_bytes += numerator_digits + denominator_digits + 256
    if output_bytes > MAX_SHIFT_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.shift_scalar_output",
            message="shift-operator scalar product exceeds its serialized output budget",
        )

    result_terms = []
    for term in operator_value.terms:
        coefficient = {
            exponent: value * scalar_fraction
            for exponent, value in _decode_poly(
                term.coefficient.numerator.terms
            ).items()
        }
        coefficient = {
            exponent: value for exponent, value in coefficient.items() if value
        }
        if coefficient:
            result_terms.append(
                {
                    "exponent": term.exponent,
                    "coefficient": _encode_rf((coefficient, {0: Fraction(1)})),
                }
            )
    result = ShiftOreOperator.model_validate({"variable": "n", "terms": result_terms})
    return ShiftOperatorScalarMultiplyResult._from_kernel(
        scalar_value, operator_value, result
    )


def shift_operator_normalize_polynomial_coefficients(
    operator: ShiftOreOperator | Mapping[str, Any],
) -> ShiftOperatorNormalizeResult:
    """Extract rational content from a polynomial-coefficient shift operator.

    The normalized operator has integer polynomial coefficients, gcd of all
    coefficient entries one, and positive leading coefficient. ``scale`` is
    the exact rational with ``operator = scale * normalized``. Rational-
    function operator coefficients are outside this normalization contract.
    """
    operator_value = _admit_polynomial_shift_operator(operator, label="operator")
    coefficients = [
        (term.exponent, _decode_poly(term.coefficient.numerator.terms))
        for term in operator_value.terms
    ]
    coefficient_count = sum(len(polynomial) for _order, polynomial in coefficients)
    if coefficient_count > MAX_SHIFT_ADDITIVE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("operator", "terms"),
            code="ore_algebra.shift_normalize_work",
            message="shift-operator normalization exceeds its coefficient-work budget",
        )
    if not coefficients:
        return ShiftOperatorNormalizeResult._from_kernel(
            operator_value,
            operator_value,
            _encode_rf(({0: Fraction(1)}, {0: Fraction(1)})),
        )

    # Build one admitted content plan. Each input rational component has at
    # most 64 digits; every lcm product fits below 192 digits before the
    # resulting scale is required to fit the 128-digit rational-function
    # carrier.
    lcm_denominators = 1
    common_numerator = 0
    flattened = [
        value for _order, polynomial in coefficients for value in polynomial.values()
    ]
    for value in flattened:
        value_numerator = abs(value.numerator)
        common_numerator = gcd(common_numerator, value_numerator)
        common = gcd(lcm_denominators, value.denominator)
        quotient = lcm_denominators // common
        if _digit_count(quotient) + _digit_count(value.denominator) > 192:
            raise OperationResourceAdmissionError(
                location=("operator", "terms"),
                code="ore_algebra.shift_normalize_intermediate_digits",
                message="normalization denominator lcm exceeds its admitted intermediate bound",
            )
        lcm_denominators = quotient * value.denominator
        if _digit_count(lcm_denominators) > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("operator", "terms"),
                code="ore_algebra.shift_normalize_scale_digits",
                message="normalization scale exceeds the rational coefficient-digit carrier",
            )
    if common_numerator == 0:
        raise RuntimeError("a nonzero operator has no nonzero polynomial coefficient")

    _leading_order, leading_polynomial = coefficients[-1]
    leading_coefficient = leading_polynomial[max(leading_polynomial)]
    sign = -1 if leading_coefficient < 0 else 1
    scale = Fraction(sign * common_numerator, lcm_denominators)
    scale_digits = max(_digit_count(scale.numerator), _digit_count(scale.denominator))
    if scale_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("scale",),
            code="ore_algebra.shift_normalize_scale_digits",
            message="normalization scale exceeds the rational coefficient-digit carrier",
        )

    output_bytes = (
        _operator_representation_byte_bound(operator_value)
        + 256
        + _digit_count(scale.numerator)
        + _digit_count(scale.denominator)
    )
    for _order, polynomial in coefficients:
        output_bytes += 256
        for value in polynomial.values():
            multiplier = lcm_denominators // value.denominator
            normalized_digits = _digit_count(value.numerator) + _digit_count(multiplier)
            if normalized_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("operator", "terms"),
                    code="ore_algebra.shift_normalize_coefficient_digits",
                    message="a primitive normalized coefficient can exceed the rational coefficient-digit carrier",
                )
            output_bytes += normalized_digits + 128
    if output_bytes > MAX_SHIFT_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.shift_normalize_output",
            message="normalized shift operator and source scale exceed the serialized output budget",
        )

    normalized_terms = []
    for exponent, polynomial in coefficients:
        normalized = {
            degree: Fraction(
                sign
                * value.numerator
                * (lcm_denominators // value.denominator)
                // common_numerator
            )
            for degree, value in polynomial.items()
        }
        normalized_terms.append(
            {
                "exponent": exponent,
                "coefficient": _encode_rf((normalized, {0: Fraction(1)})),
            }
        )
    normalized_operator = ShiftOreOperator.model_validate(
        {"variable": "n", "terms": normalized_terms}
    )
    return ShiftOperatorNormalizeResult._from_kernel(
        operator_value,
        normalized_operator,
        _encode_rf(({0: scale}, {0: Fraction(1)})),
    )


def _polynomial_evaluation_digit_bound(
    polynomial: SparseRationalPolynomial, *, max_abs_index: int
) -> tuple[int, int]:
    """Bound numerator and denominator digits of a polynomial value.

    A common denominator formed from every coefficient has at most t*D digits.
    The lifted numerator is bounded by t*D + degree*digits(index) + log10(t).
    This bounds Fraction intermediates as well as the reduced value.
    """
    if not polynomial.terms:
        return 1, 1
    term_count = len(polynomial.terms)
    max_degree = max(term.exponents[0] for term in polynomial.terms)
    coefficient_digits = max(
        max(
            _digit_count(term.coefficient.num),
            _digit_count(term.coefficient.den),
        )
        for term in polynomial.terms
    )
    index_digits = _digit_count(max(1, max_abs_index))
    numerator_digits = (
        term_count * coefficient_digits
        + max_degree * index_digits
        + _digit_count(term_count)
        + 2
    )
    denominator_digits = term_count * coefficient_digits + 1
    return numerator_digits, denominator_digits


def _rational_function_evaluation_digit_bound(
    function: RationalFunction, *, max_abs_index: int
) -> tuple[int, int]:
    numerator_n, denominator_n = _polynomial_evaluation_digit_bound(
        function.numerator, max_abs_index=max_abs_index
    )
    numerator_d, denominator_d = _polynomial_evaluation_digit_bound(
        function.denominator, max_abs_index=max_abs_index
    )
    return numerator_n + denominator_d, denominator_n + numerator_d


def _polynomial_evaluation_work_bound(
    polynomial: SparseRationalPolynomial, *, max_abs_index: int
) -> int:
    """Upper-bound digit-weighted exact work for one polynomial evaluation."""
    if not polynomial.terms:
        return 0
    term_count = len(polynomial.terms)
    max_degree = max(term.exponents[0] for term in polynomial.terms)
    coefficient_digits = max(
        max(
            _digit_count(term.coefficient.num),
            _digit_count(term.coefficient.den),
        )
        for term in polynomial.terms
    )
    index_digits = _digit_count(max(1, max_abs_index))
    return term_count * (
        term_count * coefficient_digits + max_degree * index_digits + 1
    )


def _admit_shift_prefix_output(
    operator: ShiftOreOperator,
    sequence: FiniteRationalSequence,
    start_index: int,
    residual_count: int,
    end_index: int,
) -> None:
    """Preflight exact coefficient, contribution, residual, and wire growth."""
    right_boundary_count = len(sequence.values) - residual_count
    evaluation_cells = residual_count * sum(
        len(term.coefficient.numerator.terms) + len(term.coefficient.denominator.terms)
        for term in operator.terms
    )
    if evaluation_cells > MAX_SHIFT_PREFIX_EVALUATION_CELLS:
        raise OperationResourceAdmissionError(
            location=("sequence", "values"),
            code="ore_algebra.shift_prefix_work",
            message="shift-operator prefix evaluation exceeds its coefficient-evaluation budget",
        )
    max_abs_index = max((abs(start_index + residual_count - 1), abs(end_index - 1), 1))
    work_units = residual_count * sum(
        _polynomial_evaluation_work_bound(polynomial, max_abs_index=max_abs_index)
        for term in operator.terms
        for polynomial in (term.coefficient.numerator, term.coefficient.denominator)
    )
    if work_units > MAX_SHIFT_PREFIX_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("operator", "terms"),
            code="ore_algebra.shift_prefix_work",
            message="shift-operator prefix evaluation exceeds its exact-arithmetic work budget",
        )
    coefficient_bounds = [
        _rational_function_evaluation_digit_bound(
            term.coefficient, max_abs_index=max_abs_index
        )
        for term in operator.terms
    ]
    if any(
        max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS
        for numerator_digits, denominator_digits in coefficient_bounds
    ):
        raise OperationResourceAdmissionError(
            location=("operator", "terms"),
            code="ore_algebra.shift_prefix_coefficient_digits",
            message="an evaluated shift coefficient can exceed the exact-rational result carrier",
        )

    output_bytes_bound = (
        512 * residual_count
        + 24 * right_boundary_count
        + 256 * len(operator.terms)
        + sum(
            _digit_count(value.num) + _digit_count(value.den) + 64
            for value in sequence.values
        )
        + sum(
            _digit_count(coefficient.num) + _digit_count(coefficient.den) + 64
            for term in operator.terms
            for polynomial in (term.coefficient.numerator, term.coefficient.denominator)
            for coefficient in (entry.coefficient for entry in polynomial.terms)
        )
    )
    residual_digit_bound = 1
    for offset in range(residual_count):
        contribution_bounds: list[tuple[int, int]] = []
        for term, (coefficient_num_digits, coefficient_den_digits) in zip(
            operator.terms, coefficient_bounds, strict=True
        ):
            sequence_rational = sequence.values[offset + term.exponent]
            sequence_num_digits = _digit_count(sequence_rational.num)
            sequence_den_digits = _digit_count(sequence_rational.den)
            contribution_n = coefficient_num_digits + sequence_num_digits
            contribution_d = coefficient_den_digits + sequence_den_digits
            if max(contribution_n, contribution_d) > MAX_CANONICAL_RATIONAL_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("operator", "terms", term.exponent),
                    code="ore_algebra.shift_prefix_contribution_digits",
                    message="a shift-prefix contribution can exceed the exact-rational result carrier",
                )
            contribution_bounds.append((contribution_n, contribution_d))
            output_bytes_bound += (
                coefficient_num_digits
                + coefficient_den_digits
                + sequence_num_digits
                + sequence_den_digits
                + contribution_n
                + contribution_d
                + 512
            )
        denominator_digits = sum(den for _, den in contribution_bounds)
        numerator_digits = max(
            (
                num + denominator_digits - den + _digit_count(len(contribution_bounds))
                for num, den in contribution_bounds
            ),
            default=1,
        )
        if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("operator",),
                code="ore_algebra.shift_prefix_residual_digits",
                message="the exact shift-prefix residual can exceed the rational result carrier",
            )
        residual_digit_bound = max(
            residual_digit_bound, numerator_digits, denominator_digits
        )
        output_bytes_bound += 2 * residual_digit_bound + 512
    if output_bytes_bound > MAX_SHIFT_PREFIX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("sequence", "values"),
            code="ore_algebra.shift_prefix_output",
            message="shift-prefix residual output exceeds its byte budget",
        )


def shift_operator_apply_to_sequence_prefix(
    operator: ShiftOreOperator | Mapping[str, Any],
    start_index: int,
    sequence: FiniteRationalSequence | Mapping[str, Any],
) -> ShiftOperatorPrefixResult:
    """Return exact P f residuals on a finite prefix, without global claims.

    The finite sequence has no implicit mathematical origin. ``start_index``
    binds its first stored value to that integer. Rows requiring values past
    the supplied prefix are reported separately; coefficient poles exclude
    their entire index row.
    """
    operator_value = _as_operator(operator)
    operator_value = _admit_shift_operator(operator_value, label="operator")
    try:
        sequence_value = FiniteRationalSequence.model_validate(
            sequence.model_dump()
            if isinstance(sequence, FiniteRationalSequence)
            else sequence
        )
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise ValueError("start_index must be an integer")
        end_index = start_index + len(sequence_value.values)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("sequence",),
            code="ore_algebra.shift_prefix_source",
            message="the source must be a canonical rational prefix with bounded explicit indices",
        ) from exc
    if (
        abs(start_index) > MAX_SHIFT_PREFIX_INDEX
        or abs(end_index) > MAX_SHIFT_PREFIX_INDEX
    ):
        raise OperationResourceAdmissionError(
            location=("start_index",),
            code="ore_algebra.shift_prefix_index",
            message="prefix indices exceed the admitted integer-index envelope",
        )

    max_order = operator_value.order
    residual_count = (
        len(sequence_value.values)
        if max_order < 0
        else max(0, len(sequence_value.values) - max_order)
    )
    _admit_shift_prefix_output(
        operator_value, sequence_value, start_index, residual_count, end_index
    )
    right_boundary_indices = tuple(range(start_index + residual_count, end_index))

    rows: list[ShiftOperatorPrefixResidual] = []
    exclusions: list[ShiftOperatorPrefixPoleExclusion] = []
    for offset in range(residual_count):
        index = start_index + offset
        contributions: list[ShiftOperatorPrefixContribution] = []
        pole_exponents: list[int] = []
        total = Fraction(0)
        for term in operator_value.terms:
            coefficient_numerator = _evaluate_rational_polynomial_at(
                term.coefficient.numerator, index
            )
            coefficient_denominator = _evaluate_rational_polynomial_at(
                term.coefficient.denominator, index
            )
            if coefficient_denominator == 0:
                pole_exponents.append(term.exponent)
                continue
            coefficient = coefficient_numerator / coefficient_denominator
            sequence_rational = sequence_value.values[offset + term.exponent]
            sequence_term = sequence_rational.as_fraction()
            value = coefficient * sequence_term
            total += value
            contributions.append(
                ShiftOperatorPrefixContribution(
                    exponent=term.exponent,
                    sequence_index=index + term.exponent,
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    sequence_value=sequence_rational,
                    value=CanonicalRational.from_fraction(value),
                )
            )
        if pole_exponents:
            exclusions.append(
                ShiftOperatorPrefixPoleExclusion(
                    index=index, exponents=tuple(pole_exponents)
                )
            )
            continue
        rows.append(
            ShiftOperatorPrefixResidual(
                index=index,
                contributions=tuple(contributions),
                residual=CanonicalRational.from_fraction(total),
            )
        )
    return ShiftOperatorPrefixResult._from_kernel(
        operator_value,
        start_index,
        sequence_value,
        tuple(rows),
        tuple(exclusions),
        right_boundary_indices,
    )


def _preflight_recurrence_coefficients(
    request: PolynomialRecurrencePrefixRequest,
    operator: ShiftOreOperator,
    coefficients: dict[int, _Poly],
    order: int,
    coefficient_terms: int,
) -> dict[int, _Poly]:
    """Prove coefficient-height/output bounds, then clear denominators."""
    denominator_digit_bound = sum(
        _digit_count(value.denominator)
        for poly in coefficients.values()
        for value in poly.values()
    )
    max_index = max(
        abs(request.start_index),
        abs(request.start_index + request.steps + order - 1),
        1,
    )
    max_coefficient_digits = max(
        (
            _digit_count(value.numerator)
            for poly in coefficients.values()
            for value in poly.values()
        ),
        default=1,
    )
    max_degree = max(
        (degree for poly in coefficients.values() for degree in poly), default=0
    )
    c_digits = (
        denominator_digit_bound
        + max_coefficient_digits
        + max_degree * _digit_count(max_index)
        + coefficient_terms.bit_length()
        + 2
    )
    initial_height = max(
        (
            max(_digit_count(v.num), _digit_count(v.den))
            for v in request.initial_values.values
        ),
        default=1,
    )
    height = initial_height
    for _ in range(request.steps):
        height = order * height + (order + 1) * c_digits + order.bit_length() + 2
        if height > MAX_CANONICAL_RATIONAL_DIGITS:
            raise OperationResourceAdmissionError(
                location=("steps",),
                code="ore_algebra.recurrence_coefficient_growth",
                message="finite recurrence coefficient-growth bound exceeds the exact rational carrier",
            )
    index_digits = _digit_count(max(1, max_index))
    predicted_bytes = (
        _operator_representation_byte_bound(operator)
        + (order + request.steps) * (2 * height + 96 + 24 + 3 * index_digits)
        + 512
    )
    if predicted_bytes > MAX_RECURRENCE_PREFIX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("steps",),
            code="ore_algebra.recurrence_output_bytes",
            message="finite recurrence output exceeds its admitted byte budget",
        )

    common_denominator = 1
    for poly in coefficients.values():
        for value in poly.values():
            common_denominator = (
                common_denominator
                * value.denominator
                // gcd(common_denominator, value.denominator)
            )
    return {
        exponent: {degree: value * common_denominator for degree, value in poly.items()}
        for exponent, poly in coefficients.items()
    }


def polynomial_recurrence_generate_prefix(
    operator: ShiftOreOperator | Mapping[str, Any],
    start_index: int,
    initial_values: FiniteRationalSequence | Mapping[str, Any],
    steps: int,
) -> PolynomialRecurrencePrefix:
    """Generate a finite solution prefix of a polynomial-coefficient recurrence.

    The returned relation is only asserted at ``steps`` consecutive integer
    indices. No global recurrence or infinite sequence is represented.
    """
    try:
        request = PolynomialRecurrencePrefixRequest.model_validate(
            {
                "operator": operator.model_dump()
                if isinstance(operator, ShiftOreOperator)
                else operator,
                "start_index": start_index,
                "initial_values": initial_values.model_dump()
                if isinstance(initial_values, FiniteRationalSequence)
                else initial_values,
                "steps": steps,
            }
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.finite_recurrence_request",
            message="the recurrence, initial values, index range, or step count is invalid",
        ) from exc

    op = _admit_polynomial_shift_operator(request.operator, label="operator")
    order = op.order
    coefficients = {
        term.exponent: _decode_poly(term.coefficient.numerator.terms)
        for term in op.terms
    }
    coefficient_terms = sum(len(poly) for poly in coefficients.values())
    work = (
        request.steps
        * max(1, len(coefficients))
        * max(1, max((len(p) for p in coefficients.values()), default=0))
    )
    if work > MAX_RECURRENCE_PREFIX_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.recurrence_work",
            message="finite recurrence generation exceeds its admitted coefficient-evaluation work",
        )

    # Admission proves height/output limits before denominator clearing or
    # recurrence expansion.
    integer_coefficients = _preflight_recurrence_coefficients(
        request, op, coefficients, order, coefficient_terms
    )

    leading = integer_coefficients[order]

    def evaluate(poly: dict[int, Fraction], index: int) -> Fraction:
        return sum(
            (value * index**degree for degree, value in poly.items()), Fraction(0)
        )

    # Check the complete declared finite recurrence interval before generating
    # any values, so a singular leading coefficient cannot leave partial output.
    leading_values = tuple(
        evaluate(leading, request.start_index + step) for step in range(request.steps)
    )
    if any(value == 0 for value in leading_values):
        raise OperationDomainValidationError(
            location=("operator", "leading_coefficient"),
            code="ore_algebra.singular_leading_coefficient",
            message="the leading recurrence coefficient vanishes on the declared finite index interval",
        )

    values = [value.as_fraction() for value in request.initial_values.values]
    recurrence_indices = tuple(
        request.start_index + step for step in range(request.steps)
    )
    for step, index in enumerate(recurrence_indices):
        total = Fraction(0)
        for exponent, poly in integer_coefficients.items():
            if exponent == order:
                continue
            total += evaluate(poly, index) * values[step + exponent]
        value = -total / leading_values[step]
        if (
            max(_digit_count(value.numerator), _digit_count(value.denominator))
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("values", step + order),
                code="ore_algebra.recurrence_coefficient_growth",
                message="generated recurrence value exceeds the exact rational carrier",
            )
        values.append(value)
    sequence = FiniteRationalSequence.model_validate(
        {"values": [CanonicalRational.from_fraction(value) for value in values]}
    )
    return PolynomialRecurrencePrefix._from_kernel(
        op, request.start_index, recurrence_indices, sequence
    )


def _rf_derivative(value: tuple[_Poly, _Poly]) -> tuple[_Poly, _Poly]:
    numerator, denominator = value
    numerator_derivative = {e - 1: c * e for e, c in numerator.items() if e}
    denominator_derivative = {e - 1: c * e for e, c in denominator.items() if e}
    return _normalize(
        _poly_add(
            _poly_mul(numerator_derivative, denominator),
            {e: -c for e, c in _poly_mul(numerator, denominator_derivative).items()},
        ),
        _poly_mul(denominator, denominator),
    )


def _rf_bound(value: RationalFunction) -> tuple[int, int, int]:
    """Return numerator degree, denominator degree, and coefficient digits."""
    numerator, denominator = _decode_rf(value)
    digits = max(
        (
            max(
                len(str(abs(term.coefficient.as_fraction().numerator))),
                len(str(term.coefficient.as_fraction().denominator)),
            )
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
        ),
        default=1,
    )
    return _poly_degree(numerator), _poly_degree(denominator), digits


def _derivative_rf_bound(
    bound: tuple[int, int, int], order: int
) -> tuple[int, int, int]:
    """Bound every rational-function carrier reached by ``D**order``."""
    numerator_degree, denominator_degree, digits = bound
    if numerator_degree < 0:
        return -1, 0, digits
    # N'/Q - N Q'/Q**2 raises the denominator degree by Q's degree and
    # the numerator degree by at most deg(Q)-1 on each step.
    growth_steps = 1 << order
    return (
        numerator_degree + order * max(denominator_degree - 1, 0),
        denominator_degree * (order + 1),
        # Fraction additions can multiply the active denominator at every
        # derivative stage.  Exponential accounting is conservative but keeps
        # the admission independent of the eventual cancellation pattern.
        (digits + 4) * growth_steps,
    )


def _admit_differential_result_bounds(
    contributions: list[tuple[int, tuple[int, int, int], tuple[int, int, int], int]],
) -> None:
    """Admit the final RF carrier before any differential expansion.

    Each item is ``(output order, left coefficient bound, derivative bound,
    binomial-digit bound)``.  The common-denominator construction is a sound
    upper bound for both the intermediate sums and the normalized result.
    """
    if not contributions:
        return
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for exponent, left, derivative, binomial_digits in contributions:
        left_num, left_den, left_digits = left
        derivative_num, derivative_den, derivative_digits = derivative
        contribution_num = (
            -1 if left_num < 0 or derivative_num < 0 else left_num + derivative_num
        )
        contribution_den = left_den + derivative_den
        contribution_digits = left_digits + derivative_digits + binomial_digits
        groups.setdefault(exponent, []).append(
            (contribution_num, contribution_den, contribution_digits)
        )
    for exponent, values in groups.items():
        denominator_degree = sum(value[1] for value in values)
        numerator_degree = (
            max(
                value[0] + denominator_degree - value[1]
                for value in values
                if value[0] >= 0
            )
            if any(value[0] >= 0 for value in values)
            else -1
        )
        # Common-denominator lifting and final monic normalization can each
        # combine one active coefficient with the other contributions.
        coefficient_digits = 2 * sum(value[2] for value in values) + 8 * len(values)
        if (
            max(numerator_degree, denominator_degree)
            > MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
            or max(numerator_degree, denominator_degree) + 1
            > MAX_RATIONAL_FUNCTION_TERMS
            or coefficient_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("result", exponent),
                code="ore_algebra.differential_result_carrier",
                message="differential Ore output exceeds the rational-function carrier budget",
            )


def _as_differential_operator(
    value: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialOreOperator:
    try:
        return (
            value
            if isinstance(value, DifferentialOreOperator)
            else DifferentialOreOperator.model_validate(value)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.differential_operator",
            message="the differential operator must be a valid typed value",
        ) from exc


def _as_rational_function(
    value: RationalFunction | Mapping[str, Any],
) -> RationalFunction:
    try:
        return (
            value
            if isinstance(value, RationalFunction)
            else RationalFunction.model_validate(value)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("function",),
            code="ore_algebra.differential_function",
            message="the applied function must be a valid typed value",
        ) from exc


def _admit_differential_operator(
    operator: DifferentialOreOperator,
) -> DifferentialOreOperator:
    """Revalidate typed operators, including their QQ(x) parent axis."""
    try:
        value = DifferentialOreOperator.model_validate(operator.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.differential_operator",
            message="the differential operator must be canonical over QQ(x)",
        ) from exc
    for index, term in enumerate(value.terms):
        try:
            require_canonical_rational_function(
                term.coefficient,
                maximum_terms=MAX_SHIFT_COEFFICIENT_TERMS,
                maximum_exponent=MAX_SHIFT_COEFFICIENT_DEGREE,
                maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
                label=f"differential coefficient {index}",
            )
        except Exception as exc:
            raise OperationDomainValidationError(
                location=("terms", index),
                code="ore_algebra.differential_coefficient",
                message=str(exc),
            ) from exc
    return value


def differential_series_construct(
    operator: DifferentialOreOperator | Mapping[str, Any],
    initial_derivatives: FiniteRationalSequence | Mapping[str, Any],
) -> DFinitePowerSeries:
    """Bind an ordinary-point differential equation to complete initial data."""
    try:
        request = DFinitePowerSeriesRequest.model_validate(
            {
                "operator": operator.model_dump()
                if isinstance(operator, DifferentialOreOperator)
                else operator,
                "initial_derivatives": initial_derivatives.model_dump()
                if isinstance(initial_derivatives, FiniteRationalSequence)
                else initial_derivatives,
                "center": 0,
            }
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.dfinite_series_initial_value_problem",
            message=(
                "the differential equation must be nonzero, regular at x=0, "
                "and have complete initial derivatives at its ordinary center"
            ),
        ) from exc
    admitted_operator = _admit_differential_operator(request.operator)
    output_bytes = len(request.model_dump_json().encode("utf-8")) + 32
    if output_bytes > MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.dfinite_series_output_bytes",
            message="the D-finite formal-series value exceeds its serialized byte budget",
        )
    try:
        return DFinitePowerSeries.model_validate(
            {
                "operator": admitted_operator,
                "initial_derivatives": request.initial_derivatives,
                "center": 0,
            }
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.dfinite_series_initial_value_problem",
            message=(
                "the differential equation must be regular at x=0 and have a "
                "nonzero leading coefficient at its ordinary center x=0"
            ),
        ) from exc


def _admit_dfinite_series(value: DFinitePowerSeries) -> DFinitePowerSeries:
    """Re-admit a D-finite series before relying on its ODE and initial data."""
    return differential_series_construct(value.operator, value.initial_derivatives)


def _digits_for_bit_bound(bits: int) -> int:
    """Conservatively convert a positive bit bound to decimal digits."""
    return (max(bits, 1) * 30_103 + 99_999) // 100_000 + 1


def _reject_dfinite_prefix(
    *, location: tuple[str | int, ...], code: str, message: str
) -> None:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def _admit_dfinite_prefix_estimates(
    *,
    order: int,
    count: int,
    coefficient_denominators: list[int],
    coefficient_numerators: list[int],
    coefficient_terms: int,
    initial_denominator_bits: int,
    initial_numerator_bits: int,
    initial_output_digits: int,
) -> None:
    """Admit exact scalar growth, recurrence work, and serialized output."""
    denominator_bit_bound = 1 + sum(
        value.bit_length() for value in coefficient_denominators
    )
    numerator_bit_bound = max(
        (max(1, value.bit_length()) for value in coefficient_numerators), default=1
    )
    scalar_bit_bound = denominator_bit_bound + numerator_bit_bound
    if scalar_bit_bound > MAX_DFINITE_PREFIX_SCALAR_BITS:
        _reject_dfinite_prefix(
            location=("series", "operator"),
            code="ore_algebra.dfinite_prefix_scalar_bound",
            message="the common exact coefficient scale exceeds the Taylor-prefix scalar bound",
        )

    operator_bits = (
        denominator_bit_bound
        + numerator_bit_bound
        + (coefficient_terms + 1).bit_length()
        + 2
    )
    numerator_bits = initial_numerator_bits
    denominator_bits = initial_denominator_bits
    output_digits = initial_output_digits
    steps = max(0, count - order) if order else count
    work = coefficient_terms * max(1, denominator_bit_bound) ** 2 + count * (
        coefficient_terms + order + 1
    )
    for step in range(steps):
        factorial_bits = order * max(1, (step + order + 1).bit_length())
        sum_bits = (coefficient_terms + 1).bit_length() + 1
        numerator_bits += operator_bits + factorial_bits + sum_bits
        denominator_bits += operator_bits + factorial_bits + 1
        numerator_digits = _digits_for_bit_bound(numerator_bits)
        denominator_digits = _digits_for_bit_bound(denominator_bits)
        if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
            _reject_dfinite_prefix(
                location=("count",),
                code="ore_algebra.dfinite_prefix_coefficient_growth",
                message="the conservative exact Taylor-coefficient height bound exceeds the rational carrier",
            )
        output_digits += numerator_digits + denominator_digits
        scalar_cost = numerator_bits + denominator_bits + operator_bits
        work += (coefficient_terms * (order + 2) + order + 1) * scalar_cost
        if output_digits > MAX_SEQUENCE_TOTAL_DIGITS:
            _reject_dfinite_prefix(
                location=("count",),
                code="ore_algebra.dfinite_prefix_output_digits",
                message="the admitted Taylor-prefix representation exceeds the finite-sequence digit budget",
            )
        if work > MAX_DFINITE_PREFIX_WORK_UNITS:
            _reject_dfinite_prefix(
                location=("count",),
                code="ore_algebra.dfinite_prefix_work",
                message="the exact Taylor-prefix recurrence exceeds its admitted work budget",
            )

    if output_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        _reject_dfinite_prefix(
            location=("count",),
            code="ore_algebra.dfinite_prefix_output_digits",
            message="the admitted Taylor-prefix representation exceeds the finite-sequence digit budget",
        )
    if work > MAX_DFINITE_PREFIX_WORK_UNITS:
        _reject_dfinite_prefix(
            location=("count",),
            code="ore_algebra.dfinite_prefix_work",
            message="the exact Taylor-prefix recurrence exceeds its admitted work budget",
        )
    if 512 + 2 * output_digits + 96 * count > MAX_DFINITE_PREFIX_OUTPUT_BYTES:
        _reject_dfinite_prefix(
            location=("count",),
            code="ore_algebra.dfinite_prefix_output_bytes",
            message="the exact Taylor prefix exceeds its serialized-output budget",
        )


def _compute_admitted_dfinite_prefix(
    initial: tuple[Fraction, ...],
    polynomials: list[tuple[int, dict[int, Fraction]]],
    denominators: list[int],
    order: int,
    count: int,
) -> FiniteRationalSequence:
    """Clear coefficient denominators and evaluate a previously admitted series."""
    common_denominator = 1
    for denominator in denominators:
        common_denominator = lcm(common_denominator, denominator)
    scaled = [
        (
            exponent,
            {
                degree: coefficient.numerator
                * (common_denominator // coefficient.denominator)
                for degree, coefficient in polynomial.items()
            },
        )
        for exponent, polynomial in polynomials
    ]
    leading = next(polynomial for exponent, polynomial in scaled if exponent == order)
    leading_constant = leading.get(0, 0)
    if leading_constant == 0:
        raise OperationDomainValidationError(
            location=("series", "operator"),
            code="ore_algebra.dfinite_prefix_singular_center",
            message="the leading differential coefficient must be nonzero at x=0",
        )
    values = list(initial)
    steps = max(0, count - order) if order else count
    for m in range(steps):
        accumulated = Fraction(0)
        for exponent, polynomial in scaled:
            for degree, coefficient in polynomial.items():
                if (exponent == order and degree == 0) or degree > m:
                    continue
                source_index = m - degree + exponent
                if source_index >= len(values):
                    continue
                rising = 1
                for factor in range(m - degree + 1, m - degree + exponent + 1):
                    rising *= factor
                accumulated += coefficient * rising * values[source_index]
        pivot = leading_constant
        for factor in range(m + 1, m + order + 1):
            pivot *= factor
        values.append(-accumulated / pivot)
    return FiniteRationalSequence(
        values=tuple(CanonicalRational.from_fraction(value) for value in values[:count])
    )


def differential_series_generate_prefix(
    series: DFinitePowerSeries | Mapping[str, Any], count: int
) -> FiniteRationalSequence:
    """Compute the first ``count`` Taylor coefficients at the ordinary point 0.

    The current bounded arithmetic domain requires polynomial differential
    coefficients in QQ[x]. Initial data are derivatives, so input value i is
    divided by i! before it is emitted as the coefficient of x^i.
    """
    try:
        request = DFinitePowerSeriesPrefixRequest.model_validate(
            {
                "series": series.model_dump()
                if isinstance(series, DFinitePowerSeries)
                else series,
                "count": count,
            }
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.dfinite_prefix_request",
            message="the series prefix request must contain an ordinary-point D-finite value and a bounded count",
        ) from exc

    admitted = _admit_dfinite_series(request.series)
    operator = admitted.operator
    order = operator.order
    count = request.count
    if count > MAX_SEQUENCE_LENGTH:
        _reject_dfinite_prefix(
            location=("count",),
            code="ore_algebra.dfinite_prefix_count",
            message="the requested Taylor prefix exceeds the finite-sequence length limit",
        )
    if count == 0:
        return FiniteRationalSequence(values=())
    if order == 0:
        output_bytes = 512 + 96 * count
        if output_bytes > MAX_DFINITE_PREFIX_OUTPUT_BYTES:
            _reject_dfinite_prefix(
                location=("count",),
                code="ore_algebra.dfinite_prefix_output_bytes",
                message="the zero Taylor prefix exceeds its serialized-output budget",
            )
        return FiniteRationalSequence(
            values=tuple(
                CanonicalRational.from_fraction(Fraction(0)) for _ in range(count)
            )
        )

    polynomials: list[tuple[int, dict[int, Fraction]]] = []
    coefficient_denominators: list[int] = []
    coefficient_numerators: list[int] = []
    coefficient_terms = 0
    denominator_bit_bound = 1
    for term_index, term in enumerate(operator.terms):
        denominator = term.coefficient.denominator.terms
        if not (
            len(denominator) == 1
            and denominator[0].exponents == (0,)
            and denominator[0].coefficient.as_fraction() == 1
        ):
            raise OperationDomainValidationError(
                location=("series", "operator", "terms", term_index, "coefficient"),
                code="ore_algebra.dfinite_prefix_polynomial_coefficients",
                message="Taylor prefix generation currently requires polynomial coefficients in QQ[x]",
            )
        polynomial = _decode_poly(term.coefficient.numerator.terms)
        polynomials.append((term.order, polynomial))
        coefficient_terms += len(polynomial)
        for coefficient in polynomial.values():
            coefficient_denominators.append(coefficient.denominator)
            coefficient_numerators.append(abs(coefficient.numerator))
            denominator_bit_bound += coefficient.denominator.bit_length()

    derivatives = tuple(
        value.as_fraction() for value in admitted.initial_derivatives.values
    )
    initial = tuple(
        derivative / factorial(index) for index, derivative in enumerate(derivatives)
    )
    initial_denominator_bits = (
        sum(
            derivative.denominator.bit_length() + factorial(index).bit_length()
            for index, derivative in enumerate(derivatives)
        )
        + 1
    )
    initial_numerator_bits = initial_denominator_bits + max(
        (max(1, abs(derivative.numerator).bit_length()) for derivative in derivatives),
        default=1,
    )
    initial_digits = 0
    for index, value in enumerate(initial[:count]):
        numerator_digits = _digits_for_bit_bound(
            max(1, abs(value.numerator).bit_length())
        )
        denominator_digits = _digits_for_bit_bound(value.denominator.bit_length())
        if max(numerator_digits, denominator_digits) > MAX_CANONICAL_RATIONAL_DIGITS:
            _reject_dfinite_prefix(
                location=("series", "initial_derivatives", index),
                code="ore_algebra.dfinite_prefix_coefficient_growth",
                message="an initial Taylor coefficient exceeds the rational carrier",
            )
        initial_digits += numerator_digits + denominator_digits

    _admit_dfinite_prefix_estimates(
        order=order,
        count=count,
        coefficient_denominators=coefficient_denominators,
        coefficient_numerators=coefficient_numerators,
        coefficient_terms=coefficient_terms,
        initial_denominator_bits=initial_denominator_bits,
        initial_numerator_bits=initial_numerator_bits,
        initial_output_digits=initial_digits,
    )
    return _compute_admitted_dfinite_prefix(
        initial, polynomials, coefficient_denominators, order, count
    )


def _differential_coefficient_input(
    operator: DifferentialOreOperator | Mapping[str, Any],
) -> tuple[DifferentialOreOperator, list[tuple[int, _Poly]], int, int, int]:
    """Canonicalize the ODE and collect its bounded polynomial coefficients."""
    try:
        value = (
            operator
            if isinstance(operator, DifferentialOreOperator)
            else DifferentialOreOperator.model_validate(operator)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.coefficient_recurrence_operator",
            message="the input must be a valid differential Ore operator over QQ(x)",
        ) from exc
    value = _admit_differential_operator(value)

    polynomials: list[tuple[int, _Poly]] = []
    maximum_degree = 0
    work_bound = 0
    byte_bound = 512
    for term_index, term in enumerate(value.terms):
        denominator = term.coefficient.denominator.terms
        if not (
            len(denominator) == 1
            and denominator[0].exponents == (0,)
            and denominator[0].coefficient.as_fraction() == 1
        ):
            raise OperationDomainValidationError(
                location=("operator", "terms", term_index, "coefficient"),
                code="ore_algebra.coefficient_recurrence_polynomial_domain",
                message="coefficient recurrence conversion requires polynomial coefficients in QQ[x]",
            )
        polynomial = _decode_poly(term.coefficient.numerator.terms)
        maximum_degree = max(maximum_degree, max(polynomial))
        work_bound += len(polynomial) * (term.order + 1) ** 2
        byte_bound += 384 + sum(
            192
            + len(str(abs(coefficient.numerator)))
            + len(str(coefficient.denominator))
            for coefficient in polynomial.values()
        )
        polynomials.append((term.order, polynomial))
    if not value.terms:
        raise OperationDomainValidationError(
            location=("operator",),
            code="ore_algebra.coefficient_recurrence_zero_operator",
            message="the zero differential operator has no coefficient recurrence",
        )
    boundary_work = maximum_degree * sum(
        len(polynomial) * (order + 1) for order, polynomial in polynomials
    )
    boundary_work += maximum_degree * (MAX_SHIFT_ORDER + 1) * 8
    return value, polynomials, maximum_degree, work_bound + boundary_work, byte_bound


def _coefficient_recurrence_boundary_rows(
    polynomials: list[tuple[int, _Poly]], maximum_degree: int
) -> tuple[CoefficientRecurrenceBoundaryRow, ...]:
    """Construct the finite coefficient equations before all terms are active."""
    rows = []
    for degree in range(maximum_degree):
        coefficients: dict[int, Fraction] = {}
        for order, polynomial in polynomials:
            for coefficient_degree, scalar in polynomial.items():
                if degree < coefficient_degree:
                    continue
                source_index = degree - coefficient_degree + order
                falling = 1
                for factor in range(source_index - order + 1, source_index + 1):
                    falling *= factor
                coefficients[source_index] = (
                    coefficients.get(source_index, Fraction(0)) + scalar * falling
                )
        rows.append(
            CoefficientRecurrenceBoundaryRow(
                degree=degree,
                terms=tuple(
                    CoefficientRecurrenceBoundaryTerm(
                        index=index,
                        coefficient=CanonicalRational.from_fraction(coefficient),
                    )
                    for index, coefficient in sorted(coefficients.items())
                    if coefficient
                ),
            )
        )
    return tuple(rows)


def differential_operator_to_coefficient_recurrence(
    operator: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialCoefficientRecurrence:
    """Return exact Taylor coefficient equations for a polynomial ODE.

    For ``L = sum_j p_j(x) D^j`` and ``f = sum_k a_k x^k``, the coefficient
    of ``x^m`` in ``Lf`` is computed from
    ``[x^l]p_j * (m-l+j)_falling_j * a_(m-l+j)``. Rows below the largest
    coefficient degree are returned separately because some summands have
    not entered the coefficient extraction range yet.
    """
    value, polynomials, maximum_degree, work_bound, byte_bound = (
        _differential_coefficient_input(operator)
    )
    if work_bound > MAX_COEFFICIENT_RECURRENCE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.coefficient_recurrence_work",
            message="coefficient recurrence expansion exceeds its admitted work budget",
        )
    slopes = [
        order - degree for order, polynomial in polynomials for degree in polynomial
    ]
    minimum_slope = min(slopes)
    maximum_shift = max(slopes) - minimum_slope
    if maximum_shift > MAX_SHIFT_ORDER:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.coefficient_recurrence_shift",
            message="the recurrence shift span exceeds its typed output bound",
        )
    input_values = [
        coefficient
        for _order, polynomial in polynomials
        for coefficient in polynomial.values()
    ]
    common_denominator_bits = sum(
        value.denominator.bit_length() for value in input_values
    )
    maximum_numerator_bits = max(
        (abs(value.numerator).bit_length() for value in input_values), default=1
    )
    output_numerator_digits = _digits_for_bit_bound(
        maximum_numerator_bits + common_denominator_bits + 8 * value.order + 18
    )
    output_denominator_digits = _digits_for_bit_bound(common_denominator_bits)
    # The returned recurrence is itself a ShiftOreOperator consumed by the
    # shift-operation envelope; admit its generated coefficients against that
    # same bound before expanding falling factorials.
    if (
        max(output_numerator_digits, output_denominator_digits)
        > MAX_SHIFT_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("operator", "terms"),
            code="ore_algebra.coefficient_recurrence_coefficient_digits",
            message="a coefficient recurrence coefficient may exceed the exact rational carrier",
        )
    coefficient_digits = max(output_numerator_digits, output_denominator_digits)
    projected_output_bytes = byte_bound
    projected_output_bytes += (MAX_SHIFT_ORDER + 1) * (
        256 + 17 * (2 * coefficient_digits + 128)
    )
    projected_output_bytes += maximum_degree * 256
    projected_output_bytes += (
        maximum_degree * (MAX_SHIFT_ORDER + 1) * (2 * coefficient_digits + 128)
    )
    if projected_output_bytes > MAX_COEFFICIENT_RECURRENCE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.coefficient_recurrence_output",
            message="coefficient recurrence exceeds its serialized output budget",
        )
    # The largest possible product is bounded by the input term count times
    # the falling-factorial degree. Admit that construction before expansion.
    recurrence_polynomials: dict[int, _Poly] = {}
    for order, polynomial in polynomials:
        for degree, scalar in polynomial.items():
            shift = order - degree - minimum_slope
            falling = _falling_factorial_polynomial(shift, order)
            contribution = {
                power: scalar * coefficient for power, coefficient in falling.items()
            }
            target = recurrence_polynomials.setdefault(shift, {})
            for power, coefficient in contribution.items():
                target[power] = target.get(power, Fraction(0)) + coefficient
                if target[power] == 0:
                    del target[power]

    recurrence_operator = ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {
                    "exponent": shift,
                    "coefficient": _encode_rf((polynomial, {0: Fraction(1)}), "n"),
                }
                for shift, polynomial in sorted(recurrence_polynomials.items())
                if polynomial
            ],
        }
    )
    boundary_rows = _coefficient_recurrence_boundary_rows(polynomials, maximum_degree)
    return DifferentialCoefficientRecurrence(
        operator=value,
        recurrence=recurrence_operator,
        valid_from=maximum_degree + minimum_slope,
        boundary_rows=boundary_rows,
    )


def _admit_differential_function(value: RationalFunction) -> RationalFunction:
    """Re-admit the relied-on QQ(x) carrier and its operation envelope."""
    try:
        canonical = RationalFunction.model_validate(value.model_dump())
        if canonical.variables != ("x",):
            raise ValueError("the function must use the QQ(x) axis")
        require_canonical_rational_function(
            canonical,
            maximum_terms=MAX_SHIFT_COEFFICIENT_TERMS,
            maximum_exponent=MAX_SHIFT_COEFFICIENT_DEGREE,
            maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
            label="differential function",
        )
        return canonical
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("function",),
            code="ore_algebra.differential_function",
            message="the applied function must be canonical over QQ(x)",
        ) from exc


def _preflight_differential_addition(
    left: DifferentialOreOperator,
    right: DifferentialOreOperator,
) -> tuple[tuple[int, RationalFunction | None, RationalFunction | None], ...]:
    """Bound sparse rational-function sums before any coefficient expansion."""
    left_by_order = {term.order: term.coefficient for term in left.terms}
    right_by_order = {term.order: term.coefficient for term in right.terms}
    orders = tuple(sorted(left_by_order.keys() | right_by_order.keys()))
    if len(orders) > MAX_DIFFERENTIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("sum", "terms"),
            code="ore_algebra.differential_addition_terms",
            message="differential operator sum exceeds the sparse term budget",
        )

    plan = tuple(
        (order, left_by_order.get(order), right_by_order.get(order)) for order in orders
    )
    work = len(left.terms) + len(right.terms)
    output_bytes = 512 + len(left.model_dump_json()) + len(right.model_dump_json())
    for order, first, second in plan:
        if first is None or second is None:
            coefficient = first or second
            assert coefficient is not None
            numerator_degree, denominator_degree, digits = _rf_bound(coefficient)
            numerator_terms = len(coefficient.numerator.terms)
            denominator_terms = len(coefficient.denominator.terms)
        else:
            first_bound = _rf_bound(first)
            second_bound = _rf_bound(second)
            first_num_terms = len(first.numerator.terms)
            first_den_terms = len(first.denominator.terms)
            second_num_terms = len(second.numerator.terms)
            second_den_terms = len(second.denominator.terms)
            # N1*D2 + N2*D1 over D1*D2. Count all sparse convolution
            # pairs before executing any of those products.
            work += (
                first_num_terms * second_den_terms
                + second_num_terms * first_den_terms
                + first_den_terms * second_den_terms
            )
            first_num_degree, first_den_degree, first_digits = first_bound
            second_num_degree, second_den_degree, second_digits = second_bound
            numerator_degree = max(
                first_num_degree + second_den_degree,
                second_num_degree + first_den_degree,
            )
            denominator_degree = first_den_degree + second_den_degree
            # Account for cross-product carries and coefficient growth during
            # exact polynomial-GCD normalization, not just raw convolution.
            digits = 2 * (first_digits + second_digits + 2) + 16
            numerator_terms = min(
                first_num_terms * second_den_terms + second_num_terms * first_den_terms,
                numerator_degree + 1,
            )
            denominator_terms = min(
                first_den_terms * second_den_terms,
                denominator_degree + 1,
            )

        if (
            max(numerator_degree, denominator_degree)
            > MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
            or numerator_terms > MAX_RATIONAL_FUNCTION_TERMS
            or denominator_terms > MAX_RATIONAL_FUNCTION_TERMS
            or digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("sum", order),
                code="ore_algebra.differential_addition_coefficient",
                message="differential operator sum exceeds the rational-function carrier budget",
            )

        # Polynomial GCD/normalization is bounded by the admitted degree and
        # scalar height as well as by sparse cross-product work.
        degree = max(numerator_degree, denominator_degree)
        work += (degree + 1) ** 2 * digits
        output_bytes += 256 + (numerator_terms + denominator_terms) * (96 + 2 * digits)

    if work > MAX_DIFFERENTIAL_ADDITIVE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("sum",),
            code="ore_algebra.differential_addition_work",
            message="differential operator addition exceeds its exact-work budget",
        )
    if output_bytes > MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("sum",),
            code="ore_algebra.differential_addition_output",
            message="differential operator sum exceeds its serialized-output budget",
        )
    return plan


def differential_operator_add(
    left: DifferentialOreOperator | Mapping[str, Any],
    right: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialOperatorAddResult:
    """Add differential Ore operators coefficientwise in left form."""
    left_value = _admit_differential_operator(_as_differential_operator(left))
    right_value = _admit_differential_operator(_as_differential_operator(right))
    plan = _preflight_differential_addition(left_value, right_value)
    terms = []
    for order, first, second in plan:
        if first is None:
            coefficient = second
        elif second is None:
            coefficient = first
        else:
            coefficient = _encode_differential_rf(
                _rf_add(_decode_rf(first), _decode_rf(second))
            )
        assert coefficient is not None
        if coefficient.numerator.terms:
            terms.append({"order": order, "coefficient": coefficient})
    result = DifferentialOreOperator.model_validate({"variable": "x", "terms": terms})
    return DifferentialOperatorAddResult._from_kernel(left_value, right_value, result)


def differential_operator_apply(
    operator: DifferentialOreOperator | Mapping[str, Any],
    function: RationalFunction | Mapping[str, Any],
) -> DifferentialOperatorApplyResult:
    """Apply a left-coefficient differential Ore operator to a QQ(x) function."""
    operator_value = _as_differential_operator(operator)
    function_value = _as_rational_function(function)
    operator_value = _admit_differential_operator(operator_value)
    function_value = _admit_differential_function(function_value)
    if function_value.variables != ("x",):
        raise OperationDomainValidationError(
            location=("function",),
            code="ore_algebra.differential_function",
            message="the applied function must use the QQ(x) axis",
        )
    function_bound = _rf_bound(function_value)
    _admit_differential_result_bounds(
        [
            (
                0,
                _rf_bound(term.coefficient),
                _derivative_rf_bound(function_bound, term.order),
                0,
            )
            for term in operator_value.terms
        ]
    )
    current = _decode_rf(function_value)
    total: tuple[_Poly, _Poly] = ({}, {0: Fraction(1)})
    for term in operator_value.terms:
        derivative = current
        for _ in range(term.order):
            derivative = _rf_derivative(derivative)
        contribution = _rf_mul(_decode_rf(term.coefficient), derivative)
        total = _rf_add(total, contribution)
    return DifferentialOperatorApplyResult._from_kernel(
        operator_value, function_value, _encode_differential_rf(total)
    )


def differential_operator_multiply(
    left: DifferentialOreOperator | Mapping[str, Any],
    right: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialOperatorMultiplyResult:
    """Multiply left-coefficient differential Ore operators, D a=aD+a'."""
    left_value = _as_differential_operator(left)
    right_value = _as_differential_operator(right)
    left_value = _admit_differential_operator(left_value)
    right_value = _admit_differential_operator(right_value)
    if left_value.order + right_value.order > MAX_DIFFERENTIAL_ORDER:
        raise OperationResourceAdmissionError(
            location=("left",),
            code="ore_algebra.differential_order",
            message="differential product order exceeds the admitted envelope",
        )
    right_bounds = {
        term.order: _rf_bound(term.coefficient) for term in right_value.terms
    }
    _admit_differential_result_bounds(
        [
            (
                first.order - k + second.order,
                _rf_bound(first.coefficient),
                _derivative_rf_bound(right_bounds[second.order], k),
                len(str(comb(first.order, k))),
            )
            for first in left_value.terms
            for second in right_value.terms
            for k in range(first.order + 1)
        ]
    )
    accumulated: dict[int, tuple[_Poly, _Poly]] = {}
    for first in left_value.terms:
        coefficient = _decode_rf(first.coefficient)
        for second in right_value.terms:
            derivative = _decode_rf(second.coefficient)
            for k in range(first.order + 1):
                if k:
                    derivative = _rf_derivative(derivative)
                if not derivative[0]:
                    continue
                contribution = _rf_mul(coefficient, derivative)
                exponent = first.order - k + second.order
                if exponent > MAX_DIFFERENTIAL_ORDER:
                    raise OperationResourceAdmissionError(
                        location=("product",),
                        code="ore_algebra.differential_order",
                        message="differential product order exceeds the admitted envelope",
                    )
                scaled = (
                    {
                        e: value * comb(first.order, k)
                        for e, value in contribution[0].items()
                    },
                    contribution[1],
                )
                if exponent in accumulated:
                    accumulated[exponent] = _rf_add(accumulated[exponent], scaled)
                else:
                    accumulated[exponent] = scaled
    terms = []
    for exponent, value in sorted(accumulated.items()):
        if value[0]:
            terms.append(
                {"order": exponent, "coefficient": _encode_differential_rf(value)}
            )
    if len(terms) > MAX_DIFFERENTIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("product",),
            code="ore_algebra.differential_terms",
            message="differential product has too many terms",
        )
    product = DifferentialOreOperator.model_validate({"variable": "x", "terms": terms})
    return DifferentialOperatorMultiplyResult._from_kernel(
        left_value, right_value, product
    )


def differential_operator_normalize_polynomial_coefficients(
    operator: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialOperatorNormalizeResult:
    """Extract rational content from an operator in QQ[x]<D>.

    The returned operator has integer polynomial coefficients with joint gcd
    one and positive leading coefficient (greatest derivative order, then
    greatest polynomial degree). ``operator = scale * normalized`` exactly.
    Rational-function coefficients are outside this operation's domain.
    """
    value = _admit_differential_operator(_as_differential_operator(operator))
    polynomials: list[tuple[int, _Poly]] = []
    flattened: list[Fraction] = []
    for term in value.terms:
        numerator, denominator = _decode_rf(term.coefficient)
        if denominator != {0: Fraction(1)}:
            raise OperationDomainValidationError(
                location=("operator", "terms", term.order, "coefficient"),
                code="ore_algebra.differential_polynomial_coefficients",
                message="differential normalization accepts only polynomial coefficients in QQ[x]",
            )
        polynomials.append((term.order, numerator))
        flattened.extend(numerator.values())

    if not flattened:
        one = _encode_differential_rf(({0: Fraction(1)}, {0: Fraction(1)}))
        return DifferentialOperatorNormalizeResult._from_kernel(value, value, one)

    denominator_lcm = 1
    numerator_gcd = 0
    for coefficient in flattened:
        numerator_gcd = gcd(numerator_gcd, abs(coefficient.numerator))
        common = gcd(denominator_lcm, coefficient.denominator)
        quotient = denominator_lcm // common
        if _digit_count(quotient) + _digit_count(coefficient.denominator) > 192:
            raise OperationResourceAdmissionError(
                location=("operator", "terms"),
                code="ore_algebra.differential_normalize_intermediate_digits",
                message="normalization denominator lcm exceeds its admitted intermediate bound",
            )
        denominator_lcm = quotient * coefficient.denominator
        if _digit_count(denominator_lcm) > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("operator", "terms"),
                code="ore_algebra.differential_normalize_scale_digits",
                message="normalization scale exceeds the rational coefficient-digit carrier",
            )

    leading_order, leading_polynomial = polynomials[-1]
    del leading_order
    leading = leading_polynomial[max(leading_polynomial)]
    sign = -1 if leading < 0 else 1
    scale = Fraction(sign * numerator_gcd, denominator_lcm)
    if (
        max(_digit_count(scale.numerator), _digit_count(scale.denominator))
        > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("scale",),
            code="ore_algebra.differential_normalize_scale_digits",
            message="normalization scale exceeds the rational coefficient-digit carrier",
        )

    output_bytes = 256 + _digit_count(scale.numerator) + _digit_count(scale.denominator)
    for _order, polynomial in polynomials:
        output_bytes += 256
        for coefficient in polynomial.values():
            multiplier = denominator_lcm // coefficient.denominator
            digits = _digit_count(coefficient.numerator) + _digit_count(multiplier)
            if digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("operator", "terms"),
                    code="ore_algebra.differential_normalize_coefficient_digits",
                    message="a primitive normalized coefficient can exceed the rational coefficient-digit carrier",
                )
            output_bytes += digits + 128
    if output_bytes > MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("normalized",),
            code="ore_algebra.differential_normalize_output",
            message="normalized differential operator exceeds its serialized output budget",
        )

    normalized_terms = []
    for order, polynomial in polynomials:
        normalized = {
            degree: Fraction(
                sign
                * coefficient.numerator
                * (denominator_lcm // coefficient.denominator)
                // numerator_gcd
            )
            for degree, coefficient in polynomial.items()
        }
        normalized_terms.append(
            {
                "order": order,
                "coefficient": _encode_differential_rf((normalized, {0: Fraction(1)})),
            }
        )
    normalized_operator = DifferentialOreOperator.model_validate(
        {"variable": "x", "terms": normalized_terms}
    )
    return DifferentialOperatorNormalizeResult._from_kernel(
        value,
        normalized_operator,
        _encode_differential_rf(({0: scale}, {0: Fraction(1)})),
    )


__all__ = [
    "differential_operator_add",
    "differential_operator_apply",
    "differential_operator_multiply",
    "differential_operator_normalize_polynomial_coefficients",
    "shift_operator_multiply",
    "shift_operator_power",
]
