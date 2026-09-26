"""Native exact shift Ore-operator arithmetic over QQ(n)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import comb, lcm
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    MAX_DIFFERENTIAL_ORDER,
    MAX_DIFFERENTIAL_TERMS,
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    MAX_SHIFT_LEDGER_ROWS,
    MAX_SHIFT_RESULT_DEGREE,
    MAX_SHIFT_RESULT_DIGITS,
    MAX_SHIFT_RESULT_ORDER,
    DifferentialOperatorApplyResult,
    DifferentialOperatorMultiplyResult,
    DifferentialOreOperator,
    ShiftMultiplyLedgerRow,
    ShiftOperatorMultiplyResult,
    ShiftOreOperator,
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


def _poly_product_coefficient_digits(left: _Poly, right: _Poly) -> int:
    """Bound coefficient height after sparse rational polynomial convolution."""
    if not left or not right:
        return 1

    def profile(poly: _Poly) -> tuple[int, int]:
        denominator_sum = 0
        maximum_numerator_minus_denominator = 0
        for coefficient in poly.values():
            numerator_digits = len(str(abs(coefficient.numerator)))
            denominator_digits = len(str(coefficient.denominator))
            denominator_sum += denominator_digits
            maximum_numerator_minus_denominator = max(
                maximum_numerator_minus_denominator,
                numerator_digits - denominator_digits,
            )
        return denominator_sum, maximum_numerator_minus_denominator

    left_denominators, left_height = profile(left)
    right_denominators, right_height = profile(right)
    denominator_digits = left_denominators + right_denominators
    collision_count = min(len(left), len(right))
    addition_digits = len(str(collision_count)) if collision_count > 1 else 0
    numerator_digits = (
        denominator_digits + max(0, left_height + right_height) + addition_digits
    )
    return max(denominator_digits, numerator_digits)


def _rf_product_coefficient_digits(
    left: tuple[_Poly, _Poly], right: tuple[_Poly, _Poly]
) -> int:
    return max(
        _poly_product_coefficient_digits(left[0], right[0]),
        _poly_product_coefficient_digits(left[1], right[1]),
    )


def _derivative_rf_bound(
    bound: tuple[int, int, int],
    order: int,
    *,
    source: RationalFunction | None = None,
) -> tuple[int, int, int]:
    """Bound every rational-function carrier reached by ``D**order``."""
    numerator_degree, denominator_degree, digits = bound
    if numerator_degree < 0:
        return -1, 0, digits
    # N'/Q - N Q'/Q**2 raises the denominator degree by Q's degree and
    # the numerator degree by at most deg(Q)-1 on each step.
    coefficient_digits = digits
    if order and source is not None:
        numerator, denominator = _decode_rf(source)

        def denominator_profile(poly: _Poly) -> tuple[int, int]:
            common = lcm(*(coefficient.denominator for coefficient in poly.values()))
            return _bounded_integer_digits(common), max(
                (
                    _bounded_integer_digits(abs(value.numerator))
                    for value in poly.values()
                ),
                default=1,
            )

        numerator_denominator_digits, numerator_digits = denominator_profile(numerator)
        denominator_denominator_digits, denominator_digits = denominator_profile(
            denominator
        )
        collision_digits = _bounded_integer_digits(
            min(len(numerator), len(denominator))
        )
        common_product_denominator = (
            numerator_denominator_digits + denominator_denominator_digits
        )
        differentiated_numerator = (
            common_product_denominator
            + numerator_digits
            + denominator_digits
            + _bounded_integer_digits(max(1, numerator_degree, denominator_degree))
            + collision_digits
            + 2
        )
        squared_denominator = (
            2 * denominator_denominator_digits
            + 2 * denominator_digits
            + collision_digits
            + 2
        )
        coefficient_digits = max(differentiated_numerator, squared_denominator)
        for _ in range(1, order):
            coefficient_digits = (coefficient_digits + 4) * 2
    growth_steps = 1 << order
    return (
        numerator_degree + order * max(denominator_degree - 1, 0),
        denominator_degree * (order + 1),
        # Fraction additions can multiply the active denominator at every
        # derivative stage.  Exponential accounting is conservative but keeps
        # the admission independent of the eventual cancellation pattern.
        coefficient_digits if source is not None else (digits + 4) * growth_steps,
    )


def _bounded_integer_digits(value: int) -> int:
    """Cheap upper bound for decimal digits without converting a large int."""
    if abs(value) <= 1:
        return 1
    return (abs(value).bit_length() * 30_103 + 99_999) // 100_000


def _admit_differential_result_bounds(
    contributions: list[
        tuple[int, tuple[int, int, int], tuple[int, int, int], int]
        | tuple[int, tuple[int, int, int], tuple[int, int, int], int, int]
    ],
    *,
    check_coefficient_digits: bool = True,
) -> None:
    """Admit the final RF carrier before any differential expansion.

    Each item is ``(output order, left coefficient bound, derivative bound,
    binomial-digit bound)``.  The common-denominator construction is a sound
    upper bound for both the intermediate sums and the normalized result.
    """
    if not contributions:
        return
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for contribution in contributions:
        exponent, left, derivative, binomial_digits = contribution[:4]
        convolution_digits = contribution[4] if len(contribution) == 5 else 0
        left_num, left_den, left_digits = left
        derivative_num, derivative_den, derivative_digits = derivative
        contribution_num = (
            -1 if left_num < 0 or derivative_num < 0 else left_num + derivative_num
        )
        contribution_den = left_den + derivative_den
        contribution_digits = max(
            left_digits + derivative_digits + binomial_digits, convolution_digits
        )
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
        # A common denominator is a product of the contribution denominators.
        # Each numerator summand uses that denominator with its own denominator
        # removed, so the total component height is bounded by the sum of the
        # contribution heights plus the digits needed to add the summands.
        coefficient_digits = sum(value[2] for value in values) + len(str(len(values)))
        if (
            max(numerator_degree, denominator_degree)
            > MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
            or max(numerator_degree, denominator_degree) + 1
            > MAX_RATIONAL_FUNCTION_TERMS
            or (
                check_coefficient_digits
                and coefficient_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
            )
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
                _derivative_rf_bound(function_bound, term.order, source=function_value),
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
        id(term): (term.coefficient, _rf_bound(term.coefficient))
        for term in right_value.terms
    }
    planned = [
        (
            first,
            second,
            k,
            _derivative_rf_bound(
                right_bounds[id(second)][1],
                k,
                source=right_bounds[id(second)][0],
            ),
        )
        for first in left_value.terms
        for second in right_value.terms
        for k in range(first.order + 1)
    ]
    static_bounds: list[
        tuple[int, tuple[int, int, int], tuple[int, int, int], int]
        | tuple[int, tuple[int, int, int], tuple[int, int, int], int, int]
    ] = [
        (
            first.order - k + second.order,
            _rf_bound(first.coefficient),
            derivative_bound,
            len(str(comb(first.order, k))),
        )
        for first, second, k, derivative_bound in planned
    ]
    # Admit derivative degrees and sparse work before constructing derivative
    # coefficients; their exact coefficient supports then make convolution
    # growth admission collision-aware.
    _admit_differential_result_bounds(static_bounds, check_coefficient_digits=False)
    if any(
        derivative_bound[2] > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS
        for _, _, _, derivative_bound in planned
    ):
        raise OperationResourceAdmissionError(
            location=("right", "terms"),
            code="ore_algebra.differential_result_carrier",
            message="differential derivative exceeds the rational-function coefficient bound",
        )
    derivative_cache: dict[tuple[int, int], tuple[_Poly, _Poly]] = {}
    for second in right_value.terms:
        derivative = _decode_rf(second.coefficient)
        maximum_order = max((first.order for first in left_value.terms), default=0)
        derivative_cache[(second.order, 0)] = derivative
        for derivative_order in range(1, maximum_order + 1):
            request_checkpoint("during differential product derivative admission")
            derivative = _rf_derivative(derivative)
            derivative_cache[(second.order, derivative_order)] = derivative
    admitted_bounds: list[
        tuple[int, tuple[int, int, int], tuple[int, int, int], int]
        | tuple[int, tuple[int, int, int], tuple[int, int, int], int, int]
    ] = [
        (
            first.order - k + second.order,
            _rf_bound(first.coefficient),
            derivative_bound,
            len(str(comb(first.order, k))),
            _rf_product_coefficient_digits(
                _decode_rf(first.coefficient), derivative_cache[(second.order, k)]
            ),
        )
        for first, second, k, derivative_bound in planned
    ]
    _admit_differential_result_bounds(admitted_bounds)
    accumulated: dict[int, tuple[_Poly, _Poly]] = {}
    for first in left_value.terms:
        coefficient = _decode_rf(first.coefficient)
        for second in right_value.terms:
            for k in range(first.order + 1):
                derivative = derivative_cache[(second.order, k)]
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


__all__ = [
    "differential_operator_apply",
    "differential_operator_multiply",
    "shift_operator_multiply",
]
