"""Bounded exact first-order LCLM over the differential Ore algebra QQ(x)<D>."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    DifferentialOreOperator,
)
from jacobian.math.ore_algebras.first_order_lclm._models import (
    FirstOrderLCLMRequest,
    FirstOrderLCLMResult,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_canonical_rational_function,
)

_INPUT_DEGREE = 4
_INPUT_TERMS = 4
_INPUT_SCALAR_DIGITS = 2
_INPUT_BYTES = 2 * 1024 * 1024
_OUTPUT_BYTES = 2 * 1024 * 1024
_WORK_CELLS = 2_000_000
_OUTPUT_COEFFICIENT_TERMS = 64
_OUTPUT_COEFFICIENT_DEGREE = 64
_OUTPUT_COEFFICIENT_DIGITS = 64


def _decode_polynomial(coefficient: RationalFunction) -> dict[int, Fraction]:
    return {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in coefficient.numerator.terms
    }


def _canonical_request(
    left: DifferentialOreOperator | Mapping[str, Any],
    right: DifferentialOreOperator | Mapping[str, Any],
) -> FirstOrderLCLMRequest:
    try:
        request = FirstOrderLCLMRequest.model_validate(
            {
                "left": left.model_dump()
                if isinstance(left, DifferentialOreOperator)
                else left,
                "right": right.model_dump()
                if isinstance(right, DifferentialOreOperator)
                else right,
            }
        )
        return FirstOrderLCLMRequest.model_validate(request.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.first_order_lclm_request",
            message="both inputs must be canonical first-order differential operators",
        ) from exc


def _admit_operator(operator: DifferentialOreOperator, label: str) -> None:
    if operator.order != 1:
        raise OperationDomainValidationError(
            location=(label,),
            code="ore_algebra.first_order_lclm_order",
            message="first-order LCLM requires differential order one",
        )
    for term in operator.terms:
        coefficient = term.coefficient
        try:
            require_canonical_rational_function(
                coefficient,
                maximum_terms=_INPUT_TERMS,
                maximum_exponent=_INPUT_DEGREE,
                maximum_coefficient_digits=_INPUT_SCALAR_DIGITS,
                label=f"{label} coefficient",
            )
        except Exception as exc:
            # Canonical recognition errors describe invalid domain values;
            # only the owner-specific explicit envelope is a resource refusal.
            if getattr(exc, "type", "").startswith("polynomial.not_coprime"):
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.first_order_lclm_noncanonical_coefficient",
                    message="rational-function coefficients must be canonical",
                ) from exc
            raise OperationResourceAdmissionError(
                location=(label, "terms", term.order, "coefficient"),
                code="ore_algebra.first_order_lclm_input_coefficient_bound",
                message=(
                    "first-order LCLM inputs require polynomial coefficients "
                    "with degree at most four, four terms, and two-digit scalars"
                ),
            ) from exc
        if len(coefficient.denominator.terms) != 1 or (
            coefficient.denominator.terms[0].exponents != (0,)
            or coefficient.denominator.terms[0].coefficient.as_fraction() != 1
        ):
            raise OperationDomainValidationError(
                location=(label, "terms", term.order, "coefficient"),
                code="ore_algebra.first_order_lclm_polynomial_domain",
                message="first-order LCLM requires polynomial coefficients in QQ[x]",
            )


def _preflight(request: FirstOrderLCLMRequest) -> None:
    encoded_bytes = len(request.model_dump_json().encode("utf-8"))
    if encoded_bytes > _INPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.first_order_lclm_input_bytes",
            message="the first-order LCLM request exceeds its serialized input bound",
        )
    degree = max(
        (
            term.exponents[0]
            for operator in (request.left, request.right)
            for value in operator.terms
            for polynomial in (
                value.coefficient.numerator,
                value.coefficient.denominator,
            )
            for term in polynomial.terms
        ),
        default=0,
    )
    scalar_terms = sum(
        len(polynomial.terms)
        for operator in (request.left, request.right)
        for value in operator.terms
        for polynomial in (value.coefficient.numerator, value.coefficient.denominator)
    )
    scalar_digits = sum(
        len(str(abs(term.coefficient.num))) + len(str(term.coefficient.den))
        for operator in (request.left, request.right)
        for value in operator.terms
        for polynomial in (value.coefficient.numerator, value.coefficient.denominator)
        for term in polynomial.terms
    )
    work = (3 * degree + 1) ** 2 * (scalar_terms + 1) ** 2 * (scalar_digits + 1)
    if work > _WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.first_order_lclm_work",
            message="first-order LCLM exceeds its exact polynomial-work bound",
        )
    # Formula coefficients have numerator degree at most 3d and denominator
    # degree at most 2d before reduction; their products with an input operator
    # have degree at most 4d. Bound the canonical result before SymPy expansion.
    output_degree = 4 * degree
    output_terms = output_degree + 1
    output_digits = 3 * _INPUT_SCALAR_DIGITS + 8 * _INPUT_TERMS + 16
    if (
        output_degree > _OUTPUT_COEFFICIENT_DEGREE
        or output_terms > _OUTPUT_COEFFICIENT_TERMS
        or output_digits > _OUTPUT_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.first_order_lclm_result_bound",
            message="the first-order common multiple may exceed its coefficient envelope",
        )
    # Two inputs, two order-one multipliers, and one order-two result each have
    # at most two/three coefficients. Every coefficient is bounded above by the
    # 64-term, 64-digit rational-function envelope checked after conversion.
    maximum_result_bytes = (
        2 * encoded_bytes
        + 16_384
        + 11
        * (256 + 2 * _OUTPUT_COEFFICIENT_TERMS * (128 + 2 * _OUTPUT_COEFFICIENT_DIGITS))
    )
    if maximum_result_bytes > _OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.first_order_lclm_output_bytes",
            message="the first-order LCLM result exceeds its serialized byte bound",
        )


def _as_sympy(
    operator: DifferentialOreOperator, symbols: tuple[Any, ...]
) -> dict[int, Any]:
    return {
        term.order: rational_function_to_sympy(term.coefficient, symbols=symbols)
        for term in operator.terms
    }


def _encode_coefficient(expression: Any) -> RationalFunction:
    try:
        value = rational_function_from_sympy(
            expression,
            ("x",),
            maximum_terms=_OUTPUT_COEFFICIENT_TERMS,
        )
        require_canonical_rational_function(
            value,
            maximum_terms=_OUTPUT_COEFFICIENT_TERMS,
            maximum_exponent=_OUTPUT_COEFFICIENT_DEGREE,
            maximum_coefficient_digits=_OUTPUT_COEFFICIENT_DIGITS,
            label="first-order LCLM output coefficient",
        )
        return value
    except Exception as exc:
        raise OperationResourceAdmissionError(
            location=("result", "coefficient"),
            code="ore_algebra.first_order_lclm_output_coefficient",
            message="the first-order common multiple exceeds its exact coefficient bound",
        ) from exc


def _operator(coefficients: dict[int, Any]) -> DifferentialOreOperator:
    from sympy import cancel

    terms = []
    for order, expression in sorted(coefficients.items()):
        normalized = cancel(expression)
        if normalized == 0:
            continue
        terms.append({"order": order, "coefficient": _encode_coefficient(normalized)})
    return DifferentialOreOperator.model_validate({"variable": "x", "terms": terms})


def _first_order_product(
    left: DifferentialOreOperator, right: DifferentialOreOperator, variable: Any
) -> dict[int, Any]:
    """Apply the Weyl rule to two operators of order at most one."""
    from sympy import cancel

    p = _as_sympy(left, (variable,))
    q = _as_sympy(right, (variable,))
    p0, p1 = p.get(0, 0), p.get(1, 0)
    q0, q1 = q.get(0, 0), q.get(1, 0)
    return {
        0: cancel(p1 * q0.diff(variable) + p0 * q0),
        1: cancel(p1 * q1.diff(variable) + p1 * q0 + p0 * q1),
        2: cancel(p1 * q1),
    }


def _constant_operator(value: int) -> DifferentialOreOperator:
    return _operator({0: value})


def differential_first_order_lclm(
    left: DifferentialOreOperator | Mapping[str, Any],
    right: DifferentialOreOperator | Mapping[str, Any],
) -> FirstOrderLCLMResult:
    r"""Return the least common left multiple of first-order operators.

    For A=a1*D+a0 and B=b1*D+b0, solve U*A=V*B with
    U=b1*D+u0 and V=a1*D+v0. The Weyl rule D*a=a*D+a' gives two
    linear equations for u0,v0; the nonzero determinant case yields order two.
    """
    request = _canonical_request(left, right)
    _admit_operator(request.left, "left")
    _admit_operator(request.right, "right")
    _preflight(request)

    from sympy import cancel

    x = symbols_for_variables(("x",))[0]
    left_coefficients = _as_sympy(request.left, (x,))
    right_coefficients = _as_sympy(request.right, (x,))
    a0 = left_coefficients.get(0, 0)
    a1 = left_coefficients[1]
    b0 = right_coefficients.get(0, 0)
    b1 = right_coefficients[1]
    delta = cancel(a0 * b1 - a1 * b0)

    if delta == 0:
        # The two order-one coefficient rows are proportional over QQ(x).
        left_multiplier = _constant_operator(1)
        right_multiplier = _operator({0: cancel(a1 / b1)})
        common = request.left
    else:
        r1 = cancel(a1 * (b1.diff(x) + b0) - b1 * (a1.diff(x) + a0))
        r0 = cancel(a1 * b0.diff(x) - b1 * a0.diff(x))
        u0 = cancel((-b0 * r1 + b1 * r0) / delta)
        v0 = cancel((-a0 * r1 + a1 * r0) / delta)
        left_multiplier = _operator({0: u0, 1: b1})
        right_multiplier = _operator({0: v0, 1: a1})
        # The admitted formula has a tighter coefficient envelope than the
        # generic multiply operation. Construct its product directly so its
        # independent conservative admission cannot reject this request.
        common = _operator(_first_order_product(left_multiplier, request.left, x))

    if len(common.model_dump_json().encode("utf-8")) > _OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.first_order_lclm_output_bytes",
            message="the first-order LCLM result exceeds its serialized byte bound",
        )
    return FirstOrderLCLMResult._from_kernel(
        request.left,
        request.right,
        left_multiplier,
        right_multiplier,
        common,
    )
