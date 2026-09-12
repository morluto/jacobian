"""Non-evaluating typed polynomial expression normalization."""

from __future__ import annotations

from collections.abc import Mapping
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
from jacobian._models import StrictModel, canonicalize_json_containers
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
    """One exact reduced rational literal in the closed expression grammar."""

    kind: Literal["LITERAL"] = "LITERAL"
    value: CanonicalRational = Field(
        description="Reduced exact rational literal; source literals have at most 128 decimal digits per component."
    )


class PolynomialVariableExpression(StrictModel):
    """One variable selected from the request's ordered variable axis."""

    kind: Literal["VARIABLE"] = "VARIABLE"
    name: PolynomialVariable


class PolynomialAdd(StrictModel):
    """A finite sum; no textual or executable syntax is accepted."""

    kind: Literal["ADD"] = "ADD"
    operands: tuple[PolynomialExpression, ...] = Field(min_length=1, max_length=64)


class PolynomialMultiply(StrictModel):
    """A finite product; no division or negative powers are accepted."""

    kind: Literal["MULTIPLY"] = "MULTIPLY"
    operands: tuple[PolynomialExpression, ...] = Field(min_length=1, max_length=64)


class PolynomialPower(StrictModel):
    """A bounded power by a nonnegative integer exponent."""

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


class PolynomialExpressionSource(StrictModel):
    coefficient_domain: Literal["ZZ", "QQ"] = Field(
        description="The coefficient ring; ZZ accepts only integral literals, QQ accepts reduced rationals."
    )
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0,
        max_length=8,
        description="The ordered variable axis used by every exponent tuple; variables must be unique.",
    )
    expression: PolynomialExpression = Field(
        description=(
            "A closed non-evaluating AST containing only LITERAL, VARIABLE, ADD, "
            "MULTIPLY, and POWER nodes. ADD and MULTIPLY have 1-64 operands; "
            "POWER has a nonnegative exponent at most 32. Admission additionally "
            "bounds the tree to 256 nodes and depth 64, support to 4096 terms, "
            "total degree to 32768, and exact intermediate work/representation."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_tree(cls, value: object) -> object:
        """Reject deep or oversized raw trees before recursive AST parsing."""

        value = canonicalize_json_containers(value)
        if isinstance(value, Mapping):
            _bound_raw_expression(value.get("expression"))
        return value

    @model_validator(mode="after")
    def require_variable_axis(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("expression variable axis must be unique")
        return self


class PolynomialExpressionNormalizeRequest(PolynomialExpressionSource):
    """Wire request model; raw JSON trees are bounded before AST parsing."""


class PolynomialExpressionNormalizeResult(StrictModel):
    source: PolynomialExpressionSource
    polynomial: RationalPolynomial


_MAX_EXPRESSION_NODES = 256
_MAX_EXPRESSION_DEPTH = 64
_MAX_EXPRESSION_WORK = 8_000_000
# This is an intrinsic exact-representation budget, not a transport setting:
# 5M decimal coefficient digits leaves headroom for bounded sparse-term and
# source scaffolding while retaining useful dense results.
_MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS = 5_000_000
# Keep the operation's exact intermediate-height envelope in bits.  This is
# intentionally below the transport carrier's decimal-digit ceiling because
# additions with many unrelated denominators otherwise build very large
# unreduced intermediates before canonicalization.
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
    support_terms: int
    degree: int
    numerator_bits: int
    denominator_bits: int
    work: int
    intermediate_digits: int
    common_denominator: int | None


def _bound_raw_expression(expression: object) -> None:
    """Bound raw AST depth and cardinality before recursive model construction."""

    stack = [(expression, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ValueError(
                f"expression depth exceeds the {_MAX_EXPRESSION_DEPTH}-node path bound"
            )
        if count > _MAX_EXPRESSION_NODES:
            raise ValueError(f"expression node count exceeds {_MAX_EXPRESSION_NODES}")
        if isinstance(node, Mapping):
            kind = node.get("kind")
            if kind in ("ADD", "MULTIPLY"):
                operands = node.get("operands")
                if isinstance(operands, (list, tuple)):
                    if len(operands) > 64:
                        raise ValueError(
                            "expression nodes may have at most 64 operands"
                        )
                    stack.extend((child, depth + 1) for child in operands)
            elif kind == "POWER":
                if "base" in node:
                    stack.append((node["base"], depth + 1))


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


def _bounded_common_product(values: tuple[int | None, ...]) -> int | None:
    """Retain a common denominator when its exact product remains bounded."""

    result = 1
    for value in values:
        if value is None:
            return None
        result *= value
        if result.bit_length() > _MAX_EXPRESSION_COEFFICIENT_BITS:
            return None
    return result


def _bounded_common_power(value: int | None, exponent: int) -> int | None:
    if value is None:
        return None
    result: int = value**exponent
    if result.bit_length() > _MAX_EXPRESSION_COEFFICIENT_BITS:
        return None
    return result


def _support_bound(variable_count: int, degree: int, path_terms: int) -> int:
    """Bound attainable monomials, rather than charging expansion paths as terms."""

    if path_terms > MAX_POLYNOMIAL_TERMS:
        path_terms = MAX_POLYNOMIAL_TERMS + 1
    if variable_count == 0 or degree == 0:
        return min(path_terms, 1)
    # C(variable_count + degree, variable_count) counts monomials of total
    # degree at most ``degree``.  The variable axis is at most eight entries,
    # so this saturating product is cheap even at the degree boundary.
    support = 1
    for index in range(1, variable_count + 1):
        support = support * (degree + index) // index
        if support > MAX_POLYNOMIAL_TERMS:
            return path_terms
    return min(path_terms, support)


def _decimal_digits_from_bits(bits: int) -> int:
    """Upper-bound decimal digits without converting a large integer."""

    if bits > _MAX_EXPRESSION_COEFFICIENT_BITS:
        return MAX_CANONICAL_RATIONAL_DIGITS + 1
    # 30103 / 100000 is just above log10(2), so this rounds conservatively.
    return (bits * 30_103 + 99_999) // 100_000 + 1


def _representation_digits(
    support_terms: int, numerator_bits: int, denominator_bits: int
) -> int:
    component_digits = _decimal_digits_from_bits(
        numerator_bits
    ) + _decimal_digits_from_bits(denominator_bits)
    return _bounded_product(
        support_terms,
        component_digits,
        _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS,
    )


def _admit_literal(value: CanonicalRational) -> None:
    try:
        require_bounded_rational(value, max_digits=128, label="literal")
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.literal_bound",
            message="expression literals exceed the admitted 128-digit source bound",
        ) from error


def _metrics(
    expression: PolynomialExpression, variable_count: int
) -> _ExpressionMetrics:
    if isinstance(expression, PolynomialLiteral):
        _admit_literal(expression.value)
        numerator_bits = max(1, abs(expression.value.num).bit_length())
        denominator_bits = (
            0 if expression.value.den == 1 else expression.value.den.bit_length()
        )
        return _ExpressionMetrics(
            nodes=1,
            terms=1,
            support_terms=1,
            degree=0,
            numerator_bits=numerator_bits,
            denominator_bits=denominator_bits,
            work=1,
            intermediate_digits=_representation_digits(
                1, numerator_bits, denominator_bits
            ),
            common_denominator=expression.value.den,
        )
    if isinstance(expression, PolynomialVariableExpression):
        return _ExpressionMetrics(
            nodes=1,
            terms=1,
            support_terms=1,
            degree=1,
            numerator_bits=1,
            denominator_bits=0,
            work=1,
            intermediate_digits=_representation_digits(1, 1, 0),
            common_denominator=1,
        )
    if isinstance(expression, PolynomialPower):
        base = _metrics(expression.base, variable_count)
        exponent = expression.exponent
        if exponent == 0:
            return _ExpressionMetrics(
                nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
                terms=1,
                support_terms=1,
                degree=0,
                numerator_bits=1,
                denominator_bits=0,
                work=base.work,
                intermediate_digits=max(base.intermediate_digits, 2),
                common_denominator=1,
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
        result_terms_support = 1
        base_terms = base.terms
        base_terms_support = base.support_terms
        base_degree = base.degree
        result_degree = 0
        remaining = exponent
        while remaining:
            if remaining & 1:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(
                        result_terms_support,
                        base_terms_support,
                        _MAX_EXPRESSION_WORK,
                    ),
                )
                result_terms = _bounded_product(
                    result_terms, base_terms, MAX_POLYNOMIAL_TERMS
                )
                result_degree = min(
                    MAX_POLYNOMIAL_EXPONENT + 1, result_degree + base_degree
                )
                result_terms_support = _support_bound(
                    variable_count, result_degree, result_terms
                )
            remaining //= 2
            if remaining:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(
                        base_terms_support,
                        base_terms_support,
                        _MAX_EXPRESSION_WORK,
                    ),
                )
                base_terms = _bounded_product(
                    base_terms, base_terms, MAX_POLYNOMIAL_TERMS
                )
                base_degree = min(MAX_POLYNOMIAL_EXPONENT + 1, base_degree * 2)
                base_terms_support = _support_bound(
                    variable_count,
                    base_degree,
                    base_terms,
                )
        degree = min(MAX_POLYNOMIAL_EXPONENT + 1, base.degree * exponent)
        support_terms = _support_bound(variable_count, degree, terms)
        common_denominator = _bounded_common_power(base.common_denominator, exponent)
        intermediate_digits = max(
            base.intermediate_digits,
            _representation_digits(support_terms, numerator_bits, denominator_bits),
        )
        return _ExpressionMetrics(
            nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
            terms=terms,
            support_terms=support_terms,
            degree=degree,
            numerator_bits=numerator_bits,
            denominator_bits=denominator_bits,
            work=work,
            intermediate_digits=intermediate_digits,
            common_denominator=common_denominator,
        )
    child_metrics = [
        _metrics(operand, variable_count) for operand in expression.operands
    ]
    nodes = min(_MAX_EXPRESSION_NODES + 1, 1 + sum(row.nodes for row in child_metrics))
    if isinstance(expression, PolynomialAdd):
        terms = _bounded_sum(
            tuple(row.terms for row in child_metrics), MAX_POLYNOMIAL_TERMS
        )
        degree = max(row.degree for row in child_metrics)
        common_denominator = child_metrics[0].common_denominator
        if not all(
            child.common_denominator == common_denominator for child in child_metrics
        ):
            common_denominator = None
        denominator_bits = (
            common_denominator.bit_length()
            if common_denominator is not None
            else _bounded_sum(
                tuple(row.denominator_bits for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
        )
        if common_denominator is not None:
            numerator_bits = max(child.numerator_bits for child in child_metrics)
        else:
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
            tuple(row.work + row.support_terms for row in child_metrics),
            _MAX_EXPRESSION_WORK,
        )
    else:
        terms = 1
        work = 0
        current_terms = 1
        current_support_terms = 1
        current_degree = 0
        for child in child_metrics:
            terms = _bounded_product(terms, child.terms, MAX_POLYNOMIAL_TERMS)
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work + child.work,
            )
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work
                + _bounded_product(
                    current_support_terms,
                    child.support_terms,
                    _MAX_EXPRESSION_WORK,
                ),
            )
            current_terms = _bounded_product(
                current_terms, child.terms, MAX_POLYNOMIAL_TERMS
            )
            current_degree = min(
                MAX_POLYNOMIAL_EXPONENT + 1, current_degree + child.degree
            )
            current_support_terms = _support_bound(
                variable_count,
                current_degree,
                current_terms,
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
        common_denominator = _bounded_common_product(
            tuple(row.common_denominator for row in child_metrics)
        )
        if common_denominator is not None:
            denominator_bits = common_denominator.bit_length()
    degree = (
        max(row.degree for row in child_metrics)
        if isinstance(expression, PolynomialAdd)
        else min(MAX_POLYNOMIAL_EXPONENT + 1, sum(row.degree for row in child_metrics))
    )
    support_terms = _support_bound(variable_count, degree, terms)
    intermediate_digits = max(
        (row.intermediate_digits for row in child_metrics),
        default=0,
    )
    intermediate_digits = max(
        intermediate_digits,
        _representation_digits(support_terms, numerator_bits, denominator_bits),
    )
    return _ExpressionMetrics(
        nodes=nodes,
        terms=terms,
        support_terms=support_terms,
        degree=degree,
        numerator_bits=numerator_bits,
        denominator_bits=denominator_bits,
        work=work,
        intermediate_digits=intermediate_digits,
        common_denominator=common_denominator,
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
    source: PolynomialExpressionSource,
) -> PolynomialExpressionNormalizeResult:
    metrics = _metrics(source.expression, len(source.variables))
    if (
        metrics.nodes > _MAX_EXPRESSION_NODES
        or metrics.support_terms > MAX_POLYNOMIAL_TERMS
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
        variable: index for index, variable in enumerate(source.variables)
    }
    zero_exp = (0,) * len(source.variables)

    def evaluate(expression: PolynomialExpression) -> dict[tuple[int, ...], Fraction]:
        if isinstance(expression, PolynomialLiteral):
            if source.coefficient_domain == "ZZ" and expression.value.den != 1:
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
            exponent_vector = [0] * len(source.variables)
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

    coefficients = evaluate(source.expression)
    polynomial = RationalPolynomial(
        variables=source.variables,
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
    return PolynomialExpressionNormalizeResult(source=source, polynomial=polynomial)


__all__ = ["normalize_polynomial_expression"]
