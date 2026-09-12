"""Non-evaluating typed polynomial expression normalization."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, log2
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
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


def _metrics(expression: PolynomialExpression) -> tuple[int, int, int, int]:
    if isinstance(expression, PolynomialLiteral):
        require_bounded_rational(expression.value, max_digits=128, label="literal")
        bits = max(
            abs(expression.value.num).bit_length(), expression.value.den.bit_length()
        )
        return 1, 1, 0, bits
    if isinstance(expression, PolynomialVariableExpression):
        return 1, 1, 1, 1
    if isinstance(expression, PolynomialPower):
        nodes, terms, degree, bits = _metrics(expression.base)
        return (
            nodes + 1,
            min(MAX_POLYNOMIAL_TERMS + 1, terms**expression.exponent),
            degree * expression.exponent,
            bits * expression.exponent,
        )
    child_metrics = [_metrics(operand) for operand in expression.operands]
    nodes = 1 + sum(row[0] for row in child_metrics)
    if isinstance(expression, PolynomialAdd):
        terms = sum(row[1] for row in child_metrics)
        degree = max(row[2] for row in child_metrics)
        # A common denominator can contain every child denominator as a
        # factor, so summing rational terms is bounded by the sum of their
        # component heights (plus carry bits), not merely the largest child.
        bits = sum(row[3] for row in child_metrics) + ceil(log2(len(child_metrics)))
    else:
        terms = 1
        for row in child_metrics:
            terms = min(MAX_POLYNOMIAL_TERMS + 1, terms * row[1])
        degree = sum(row[2] for row in child_metrics)
        bits = sum(row[3] for row in child_metrics) + ceil(log2(max(1, terms)))
    return nodes, terms, degree, bits


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
    nodes, terms, degree, bits = _metrics(request.expression)
    if (
        nodes > 256
        or terms > MAX_POLYNOMIAL_TERMS
        or degree > MAX_POLYNOMIAL_EXPONENT
        or bits > 32_768
    ):
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.expansion_bound",
            message="expression expansion exceeds the admitted node, support, degree, or coefficient-height bound",
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
