"""Native exact shift Ore-operator arithmetic over QQ(n)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    MAX_SHIFT_LEDGER_ROWS,
    MAX_SHIFT_RESULT_DEGREE,
    MAX_SHIFT_RESULT_DIGITS,
    MAX_SHIFT_RESULT_ORDER,
    ShiftMultiplyLedgerRow,
    ShiftOperatorMultiplyResult,
    ShiftOreOperator,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
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
    return (
        value
        if isinstance(value, ShiftOreOperator)
        else ShiftOreOperator.model_validate(value)
    )


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


def _admit_shift_operator(operator: ShiftOreOperator, *, label: str) -> None:
    for index, term in enumerate(operator.terms):
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


def _admit_shift_multiply(left: ShiftOreOperator, right: ShiftOreOperator) -> None:
    _admit_shift_operator(left, label="left")
    _admit_shift_operator(right, label="right")
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


def _encode_rf(value: tuple[_Poly, _Poly]) -> RationalFunction:
    numerator, denominator = value
    return RationalFunction(
        domain="QQ",
        variables=("n",),
        numerator=_encode_poly(numerator),
        denominator=_encode_poly(denominator),
    )


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
    _admit_shift_multiply(left_value, right_value)
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


__all__ = ["shift_operator_multiply"]
