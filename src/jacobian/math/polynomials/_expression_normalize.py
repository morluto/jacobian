"""Non-evaluating typed polynomial expression normalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, log2
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


class PolynomialLiteral(StrictModel):
    kind: Literal["LITERAL"] = "LITERAL"
    value: CanonicalRational


class PolynomialVariableExpression(StrictModel):
    kind: Literal["VARIABLE"] = "VARIABLE"
    name: PolynomialVariable


class PolynomialAdd(StrictModel):
    kind: Literal["ADD"] = "ADD"
    operands: tuple[PolynomialExpression, ...] = Field(min_length=1, max_length=64)


class PolynomialMultiply(StrictModel):
    kind: Literal["MULTIPLY"] = "MULTIPLY"
    operands: tuple[PolynomialExpression, ...] = Field(min_length=1, max_length=64)


class PolynomialPower(StrictModel):
    kind: Literal["POWER"] = "POWER"
    base: PolynomialExpression
    exponent: StrictInt = Field(ge=0, le=32)


type PolynomialExpression = Annotated[
    PolynomialLiteral
    | PolynomialVariableExpression
    | PolynomialAdd
    | PolynomialMultiply
    | PolynomialPower,
    Field(discriminator="kind"),
]

for model in (PolynomialAdd, PolynomialMultiply, PolynomialPower):
    model.model_rebuild(_types_namespace={"PolynomialExpression": PolynomialExpression})


class PolynomialExpressionNormalizeRequest(StrictModel):
    coefficient_domain: Literal["ZZ", "QQ"]
    variables: tuple[PolynomialVariable, ...] = Field(min_length=0, max_length=8)
    expression: PolynomialExpression

    @model_validator(mode="after")
    def require_variable_axis(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("expression variable axis must be unique")
        return self


class PolynomialExpressionNormalizeResult(StrictModel):
    source: PolynomialExpressionNormalizeRequest
    polynomial: RationalPolynomial


_MAX_EXPRESSION_NODES = 256
_MAX_EXPRESSION_WORK = 8_000_000
# This is an intrinsic exact-representation budget, not a transport setting:
# 5M decimal coefficient digits leaves headroom for bounded sparse-term and
# source scaffolding while retaining useful dense results.
_MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS = 5_000_000
_MAX_EXPRESSION_COEFFICIENT_BITS = MAX_CANONICAL_RATIONAL_DIGITS


@dataclass(frozen=True, slots=True)
class _ExpressionMetrics:
    """Conservative bounds for one expression and all of its intermediates.

    Numerator and denominator heights are tracked independently.  In
    particular, adding integral coefficients grows one numerator by a carry
    bit per addition; it does not multiply the height by the number of
    operands.  The representation and work bounds are saturated so deeply
    nested powers cannot make admission itself expensive.
    """

    nodes: int
    terms: int
    degree: int
    numerator_bits: int
    denominator_bits: int
    work: int
    intermediate_digits: int


def _bounded_sum(values: list[int] | tuple[int, ...], limit: int) -> int:
    total = 0
    for value in values:
        total += value
        if total > limit:
            return limit + 1
    return total


def _bounded_product(left: int, right: int, limit: int) -> int:
    if left > limit or right > limit or left > limit // max(1, right):
        return limit + 1
    return left * right


def _bounded_power(value: int, exponent: int, limit: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _bounded_product(result, value, limit)
        exponent //= 2
        if exponent:
            value = _bounded_product(value, value, limit)
    return result


def _decimal_digits_from_bits(bits: int) -> int:
    """Upper-bound decimal digits without converting a large integer."""

    if bits > _MAX_EXPRESSION_COEFFICIENT_BITS:
        return MAX_CANONICAL_RATIONAL_DIGITS + 1
    # 30103 / 100000 is just above log10(2), so this rounds conservatively.
    return (bits * 30_103 + 99_999) // 100_000 + 1


def _representation_digits(
    terms: int, numerator_bits: int, denominator_bits: int
) -> int:
    component_digits = _decimal_digits_from_bits(
        numerator_bits
    ) + _decimal_digits_from_bits(denominator_bits)
    return _bounded_product(
        terms,
        component_digits,
        _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS,
    )


def _metrics(expression: PolynomialExpression) -> _ExpressionMetrics:
    if isinstance(expression, PolynomialLiteral):
        require_bounded_rational(expression.value, max_digits=128, label="literal")
        numerator_bits = max(1, abs(expression.value.num).bit_length())
        denominator_bits = (
            0 if expression.value.den == 1 else expression.value.den.bit_length()
        )
        return _ExpressionMetrics(
            nodes=1,
            terms=1,
            degree=0,
            numerator_bits=numerator_bits,
            denominator_bits=denominator_bits,
            work=1,
            intermediate_digits=_representation_digits(
                1, numerator_bits, denominator_bits
            ),
        )
    if isinstance(expression, PolynomialVariableExpression):
        return _ExpressionMetrics(
            nodes=1,
            terms=1,
            degree=1,
            numerator_bits=1,
            denominator_bits=0,
            work=1,
            intermediate_digits=_representation_digits(1, 1, 0),
        )
    if isinstance(expression, PolynomialPower):
        base = _metrics(expression.base)
        exponent = expression.exponent
        if exponent == 0:
            return _ExpressionMetrics(
                nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
                terms=1,
                degree=0,
                numerator_bits=1,
                denominator_bits=0,
                work=base.work,
                intermediate_digits=max(base.intermediate_digits, 2),
            )
        terms = _bounded_power(base.terms, exponent, MAX_POLYNOMIAL_TERMS)
        product_count = max(1, terms)
        numerator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            base.numerator_bits * exponent + ceil(log2(product_count)),
        )
        denominator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            base.denominator_bits * exponent,
        )
        work = base.work
        result_terms = 1
        base_terms = base.terms
        remaining = exponent
        while remaining:
            if remaining & 1:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(result_terms, base_terms, _MAX_EXPRESSION_WORK),
                )
                result_terms = _bounded_product(
                    result_terms, base_terms, MAX_POLYNOMIAL_TERMS
                )
            remaining //= 2
            if remaining:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(base_terms, base_terms, _MAX_EXPRESSION_WORK),
                )
                base_terms = _bounded_product(
                    base_terms, base_terms, MAX_POLYNOMIAL_TERMS
                )
        intermediate_digits = max(
            base.intermediate_digits,
            _representation_digits(terms, numerator_bits, denominator_bits),
        )
        return _ExpressionMetrics(
            nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
            terms=terms,
            degree=min(MAX_POLYNOMIAL_EXPONENT + 1, base.degree * exponent),
            numerator_bits=numerator_bits,
            denominator_bits=denominator_bits,
            work=work,
            intermediate_digits=intermediate_digits,
        )
    child_metrics = [_metrics(operand) for operand in expression.operands]
    nodes = min(_MAX_EXPRESSION_NODES + 1, 1 + sum(row.nodes for row in child_metrics))
    if isinstance(expression, PolynomialAdd):
        terms = _bounded_sum(
            tuple(row.terms for row in child_metrics), MAX_POLYNOMIAL_TERMS
        )
        degree = max(row.degree for row in child_metrics)
        denominator_bits = _bounded_sum(
            tuple(row.denominator_bits for row in child_metrics),
            _MAX_EXPRESSION_COEFFICIENT_BITS,
        )
        numerator_bits = 0
        for child in child_metrics:
            if denominator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS:
                numerator_bits = _MAX_EXPRESSION_COEFFICIENT_BITS + 1
                break
            other_denominator_bits = denominator_bits - child.denominator_bits
            numerator_bits = max(
                numerator_bits,
                child.numerator_bits + other_denominator_bits,
            )
        numerator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            numerator_bits + ceil(log2(len(child_metrics))),
        )
        work = _bounded_sum(
            tuple(row.work + row.terms for row in child_metrics),
            _MAX_EXPRESSION_WORK,
        )
    else:
        terms = 1
        work = 0
        current_terms = 1
        for child in child_metrics:
            terms = _bounded_product(terms, child.terms, MAX_POLYNOMIAL_TERMS)
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work + child.work,
            )
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work
                + _bounded_product(current_terms, child.terms, _MAX_EXPRESSION_WORK),
            )
            current_terms = _bounded_product(
                current_terms, child.terms, MAX_POLYNOMIAL_TERMS
            )
        degree = min(
            MAX_POLYNOMIAL_EXPONENT + 1,
            sum(row.degree for row in child_metrics),
        )
        numerator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            _bounded_sum(
                tuple(row.numerator_bits for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
            + ceil(log2(max(1, terms))),
        )
        denominator_bits = _bounded_sum(
            tuple(row.denominator_bits for row in child_metrics),
            _MAX_EXPRESSION_COEFFICIENT_BITS,
        )
    intermediate_digits = max(
        (row.intermediate_digits for row in child_metrics),
        default=0,
    )
    intermediate_digits = max(
        intermediate_digits,
        _representation_digits(terms, numerator_bits, denominator_bits),
    )
    return _ExpressionMetrics(
        nodes=nodes,
        terms=terms,
        degree=degree,
        numerator_bits=numerator_bits,
        denominator_bits=denominator_bits,
        work=work,
        intermediate_digits=intermediate_digits,
    )


def _add(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction()) + coefficient
        if not result[exponent]:
            del result[exponent]
    return result


def _multiply(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction()) + left_coefficient * right_coefficient
            )
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def normalize_polynomial_expression(
    request: PolynomialExpressionNormalizeRequest,
) -> PolynomialExpressionNormalizeResult:
    metrics = _metrics(request.expression)
    if (
        metrics.nodes > _MAX_EXPRESSION_NODES
        or metrics.terms > MAX_POLYNOMIAL_TERMS
        or metrics.degree > MAX_POLYNOMIAL_EXPONENT
        or metrics.numerator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
        or metrics.denominator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
        or metrics.work > _MAX_EXPRESSION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.expansion_bound",
            message=(
                "expression expansion exceeds the admitted node, support, degree, "
                "coefficient-height, work, or intermediate-representation bound"
            ),
        )
    if metrics.intermediate_digits > _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.result_representation_bound",
            message=(
                "normalized coefficients exceed the admitted exact "
                "representation envelope"
            ),
        )
    variable_index = {
        variable: index for index, variable in enumerate(request.variables)
    }
    zero_exp = (0,) * len(request.variables)

    def evaluate(expression: PolynomialExpression) -> dict[tuple[int, ...], Fraction]:
        if isinstance(expression, PolynomialLiteral):
            if request.coefficient_domain == "ZZ" and expression.value.den != 1:
                raise OperationDomainValidationError(
                    location=("expression",),
                    code="polynomial.expression.nonintegral_literal",
                    message="ZZ expressions require integral literals",
                )
            value = expression.value.as_fraction()
            return {} if not value else {zero_exp: value}
        if isinstance(expression, PolynomialVariableExpression):
            if expression.name not in variable_index:
                raise OperationDomainValidationError(
                    location=("expression",),
                    code="polynomial.expression.undeclared_variable",
                    message="every expression variable must belong to the declared axis",
                )
            exponent_vector = [0] * len(request.variables)
            exponent_vector[variable_index[expression.name]] = 1
            return {tuple(exponent_vector): Fraction(1)}
        if isinstance(expression, PolynomialAdd):
            result: dict[tuple[int, ...], Fraction] = {}
            for operand in expression.operands:
                result = _add(result, evaluate(operand))
            return result
        if isinstance(expression, PolynomialMultiply):
            result = {zero_exp: Fraction(1)}
            for operand in expression.operands:
                result = _multiply(result, evaluate(operand))
            return result
        result = {zero_exp: Fraction(1)}
        base = evaluate(expression.base)
        power = expression.exponent
        while power:
            if power & 1:
                result = _multiply(result, base)
            power //= 2
            if power:
                base = _multiply(base, base)
        return result

    coefficients = evaluate(request.expression)
    polynomial = RationalPolynomial(
        variables=request.variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponent,
                )
                for exponent, coefficient in sorted(coefficients.items(), reverse=True)
            )
        ),
    )
    return PolynomialExpressionNormalizeResult(source=request, polynomial=polynomial)


__all__ = ["normalize_polynomial_expression"]
