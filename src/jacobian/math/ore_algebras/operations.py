"""Native exact shift Ore-operator arithmetic over QQ(n)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_WEIGHT,
    MAX_DIFFERENTIAL_ADDITIVE_WORK_CELLS,
    MAX_DIFFERENTIAL_ORDER,
    MAX_DIFFERENTIAL_TERMS,
    MAX_SHIFT_ADDITIVE_OUTPUT_WEIGHT,
    MAX_SHIFT_ADDITIVE_WORK_CELLS,
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    MAX_SHIFT_LEDGER_ROWS,
    MAX_SHIFT_ORDER,
    MAX_SHIFT_POWER_EXPONENT,
    MAX_SHIFT_POWER_WORK_CELLS,
    MAX_SHIFT_PREFIX_EVALUATION_CELLS,
    MAX_SHIFT_PREFIX_INDEX,
    MAX_SHIFT_PREFIX_OUTPUT_WEIGHT,
    MAX_SHIFT_PREFIX_WORK_UNITS,
    MAX_SHIFT_RESULT_DEGREE,
    MAX_SHIFT_RESULT_DIGITS,
    MAX_SHIFT_RESULT_ORDER,
    MAX_SHIFT_TERMS,
    DifferentialOperatorAddResult,
    DifferentialOperatorApplyResult,
    DifferentialOperatorMultiplyResult,
    DifferentialOperatorNormalizeResult,
    DifferentialOreOperator,
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
        stages = _finish_shift_power_stage(
            next_stage,
            stage + 1,
            final=stage == exponent - 1,
        )


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
    stage: dict[int, tuple[int, int, int, int]], stage_number: int, *, final: bool
) -> dict[int, tuple[int, int, int]]:
    """Check one predicted stage against reusable operator input bounds."""

    result: dict[int, tuple[int, int, int]] = {}
    for exponent, (degree, terms, digits, pair_count) in stage.items():
        digits += len(str(pair_count - 1)) if pair_count > 1 else 0
        terms = min(degree + 1, terms)
        degree_limit = (
            MAX_SHIFT_RESULT_DEGREE if final else MAX_SHIFT_COEFFICIENT_DEGREE
        )
        term_limit = MAX_SHIFT_TERMS if final else MAX_SHIFT_COEFFICIENT_TERMS
        digit_limit = MAX_SHIFT_RESULT_DIGITS if final else MAX_SHIFT_COEFFICIENT_DIGITS
        if degree > degree_limit or terms > term_limit or digits > digit_limit:
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
    if value.order >= 0 and value.order * exponent > MAX_SHIFT_ORDER:
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


def _operator_representation_weight_bound(operator: ShiftOreOperator) -> int:
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
    output_weight = (
        _operator_representation_weight_bound(left)
        + _operator_representation_weight_bound(right)
        + 256 * len(plan)
    )
    for _exponent, first, second in plan:
        first_polynomial: dict[int, Fraction] = {} if first is None else first
        second_polynomial: dict[int, Fraction] = {} if second is None else second
        support = first_polynomial.keys() | second_polynomial.keys()
        output_weight += 128 * len(support)
        for degree in support:
            if degree not in second_polynomial:
                sole_value = first_polynomial[degree]
                numerator_digits = _digit_count(sole_value.numerator)
                denominator_digits = _digit_count(sole_value.denominator)
            elif degree not in first_polynomial:
                sole_value = second_polynomial[degree]
                numerator_digits = _digit_count(sole_value.numerator)
                denominator_digits = _digit_count(sole_value.denominator)
            else:
                first_value = first_polynomial[degree]
                second_value = second_polynomial[degree]
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
            output_weight += numerator_digits + denominator_digits + 64
    if output_weight > MAX_SHIFT_ADDITIVE_OUTPUT_WEIGHT:
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
    try:
        scalar_value = _as_rational_function(scalar)
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
    output_weight = _operator_representation_weight_bound(operator_value) + 256
    for term in operator_value.terms:
        for entry in term.coefficient.numerator.terms:
            numerator_digits = _digit_count(entry.coefficient.num) + scalar_digits
            denominator_digits = _digit_count(entry.coefficient.den) + scalar_digits
            if (
                max(numerator_digits, denominator_digits)
                > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            ):
                raise OperationResourceAdmissionError(
                    location=("operator", "terms", term.exponent),
                    code="ore_algebra.shift_scalar_coefficient_digits",
                    message="shift-operator scalar product can exceed the rational coefficient-digit carrier",
                )
            output_weight += numerator_digits + denominator_digits + 256
    if output_weight > MAX_SHIFT_ADDITIVE_OUTPUT_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.shift_scalar_output",
            message="shift-operator scalar product exceeds its serialized output budget",
        )

    result_terms: list[dict[str, Any]] = []
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

    output_weight = (
        _operator_representation_weight_bound(operator_value)
        + 256
        + _digit_count(scale.numerator)
        + _digit_count(scale.denominator)
    )
    for _order, polynomial in coefficients:
        output_weight += 256
        for value in polynomial.values():
            multiplier = lcm_denominators // value.denominator
            normalized_numerator = value.numerator * multiplier // common_numerator
            normalized_digits = _digit_count(normalized_numerator)
            if normalized_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("operator", "terms"),
                    code="ore_algebra.shift_normalize_coefficient_digits",
                    message="a primitive normalized coefficient can exceed the rational coefficient-digit carrier",
                )
            output_weight += normalized_digits + 128
    if output_weight > MAX_SHIFT_ADDITIVE_OUTPUT_WEIGHT:
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

    output_weight_bound = (
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
            output_weight_bound += (
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
        output_weight_bound += 2 * residual_digit_bound + 512
    if output_weight_bound > MAX_SHIFT_PREFIX_OUTPUT_WEIGHT:
        raise OperationResourceAdmissionError(
            location=("sequence", "values"),
            code="ore_algebra.shift_prefix_output",
            message="shift-prefix residual output exceeds its representation-weight budget",
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
    if not isinstance(start_index, int) or isinstance(start_index, bool):
        raise OperationDomainValidationError(
            location=("start_index",),
            code="ore_algebra.shift_prefix_start_index",
            message="start_index must be an integer",
        )
    try:
        sequence_value = FiniteRationalSequence.model_validate(
            sequence.model_dump()
            if isinstance(sequence, FiniteRationalSequence)
            else sequence
        )
        end_index = start_index + len(sequence_value.values)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("sequence",),
            code="ore_algebra.shift_prefix_source",
            message="the source must be a canonical rational prefix with bounded explicit indices",
        ) from exc
    last_stored_index = end_index - 1 if sequence_value.values else end_index
    if (
        abs(start_index) > MAX_SHIFT_PREFIX_INDEX
        or abs(last_stored_index) > MAX_SHIFT_PREFIX_INDEX
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


def _differential_operator_representation_weight(
    operator: DifferentialOreOperator,
) -> int:
    return 256 * len(operator.terms) + sum(
        128
        + sum(
            128
            + _digit_count(entry.coefficient.num)
            + _digit_count(entry.coefficient.den)
            for entry in polynomial.terms
        )
        for term in operator.terms
        for polynomial in (term.coefficient.numerator, term.coefficient.denominator)
    )


def _preflight_differential_addition(
    left: DifferentialOreOperator,
    right: DifferentialOreOperator,
) -> tuple[
    tuple[int, RationalFunction | tuple[RationalFunction, RationalFunction]], ...
]:
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

    plan_entries: list[
        tuple[int, RationalFunction | tuple[RationalFunction, RationalFunction]]
    ] = []
    for order in orders:
        if order in left_by_order:
            if order in right_by_order:
                plan_entries.append(
                    (order, (left_by_order[order], right_by_order[order]))
                )
            else:
                plan_entries.append((order, left_by_order[order]))
        else:
            plan_entries.append((order, right_by_order[order]))
    plan = tuple(plan_entries)
    work = len(left.terms) + len(right.terms)
    output_weight = (
        512
        + _differential_operator_representation_weight(left)
        + _differential_operator_representation_weight(right)
    )
    for order, operands in plan:
        if not isinstance(operands, tuple):
            coefficient = operands
            numerator_degree, denominator_degree, digits = _rf_bound(coefficient)
            numerator_terms = len(coefficient.numerator.terms)
            denominator_terms = len(coefficient.denominator.terms)
        else:
            first, second = operands
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
        output_weight += 256 + (numerator_terms + denominator_terms) * (96 + 2 * digits)

    if work > MAX_DIFFERENTIAL_ADDITIVE_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("sum",),
            code="ore_algebra.differential_addition_work",
            message="differential operator addition exceeds its exact-work budget",
        )
    if output_weight > MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_WEIGHT:
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
    for order, operands in plan:
        if isinstance(operands, tuple):
            coefficient = _encode_differential_rf(
                _rf_add(_decode_rf(operands[0]), _decode_rf(operands[1]))
            )
        else:
            coefficient = operands
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

    output_weight = (
        256 + _digit_count(scale.numerator) + _digit_count(scale.denominator)
    )
    for _order, polynomial in polynomials:
        output_weight += 256
        for coefficient in polynomial.values():
            multiplier = denominator_lcm // coefficient.denominator
            normalized_numerator = coefficient.numerator * multiplier // numerator_gcd
            digits = _digit_count(normalized_numerator)
            if digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("operator", "terms"),
                    code="ore_algebra.differential_normalize_coefficient_digits",
                    message="a primitive normalized coefficient can exceed the rational coefficient-digit carrier",
                )
            output_weight += digits + 128
    if output_weight > MAX_DIFFERENTIAL_ADDITIVE_OUTPUT_WEIGHT:
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
