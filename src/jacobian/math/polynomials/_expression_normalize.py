"""Non-evaluating typed polynomial expression normalization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
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

    ``support`` bounds the monomials that can be materialized, while
    ``expansion_terms`` is the larger expansion-path bound used to charge
    convolution work.  Numerator heights are measured over the common
    denominator in ``denominator``.  ``maximum_*_bits`` retain the greatest
    scalar heights seen in intermediates, including children that cancel to
    zero before a parent combines them.  The representation and work bounds
    are saturated so deeply nested powers cannot make admission itself
    expensive.
    """

    nodes: int
    support: int
    expansion_terms: int
    degree: int
    variables: frozenset[str]
    numerator_bits: int
    denominator: int | None
    zero: bool
    constant: Fraction | None
    monomial: frozenset[tuple[str, int]] | None
    total_coefficient_digits: int
    maximum_numerator_bits: int
    maximum_denominator_bits: int
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


def _ceil_log2(value: int) -> int:
    return 0 if value <= 1 else (value - 1).bit_length()


def _bounded_denominator_product(left: int | None, right: int | None) -> int | None:
    """Multiply common denominators while retaining only admitted values."""

    if left is None or right is None:
        return None
    if left == 1:
        return right
    if right == 1:
        return left
    # The product's bit length is at least the sum minus one.  Avoid creating
    # a denominator that is already outside the height envelope.
    if left.bit_length() + right.bit_length() > _MAX_EXPRESSION_COEFFICIENT_BITS + 1:
        return None
    product = left * right
    return product if product.bit_length() <= _MAX_EXPRESSION_COEFFICIENT_BITS else None


def _bounded_denominator_power(value: int | None, exponent: int) -> int | None:
    result: int | None = 1
    while exponent:
        if exponent & 1:
            result = _bounded_denominator_product(result, value)
        exponent //= 2
        if exponent:
            value = _bounded_denominator_product(value, value)
    return result


def _bounded_lcm(left: int | None, right: int | None) -> int | None:
    """Return an exact lcm when it fits the common-denominator envelope."""

    if left is None or right is None:
        return None
    if left == 1:
        return right
    if right == 1:
        return left
    return _bounded_denominator_product(left, right // gcd(left, right))


def _bounded_denominator_lcm(
    metrics: list[_ExpressionMetrics],
    *,
    skip_zero: bool = False,
) -> int | None:
    """Return a bounded common denominator for nonzero output metrics."""

    denominator: int | None = 1
    for metric in metrics:
        if skip_zero and metric.zero:
            continue
        denominator = _bounded_lcm(denominator, metric.denominator)
        if denominator is None:
            return None
    return denominator


def _denominator_bits(denominator: int | None) -> int:
    return (
        denominator.bit_length()
        if denominator is not None
        else _MAX_EXPRESSION_COEFFICIENT_BITS + 1
    )


def _bit_growth(value: int) -> int:
    """Bound the extra bits from multiplying by a positive integer."""

    return 0 if value <= 1 else value.bit_length()


def _bounded_monomial_support(degree: int, variable_count: int) -> int:
    """Bound monomials of total degree at most ``degree`` on the active axes."""

    if degree > MAX_POLYNOMIAL_EXPONENT:
        return MAX_POLYNOMIAL_TERMS + 1
    steps = min(degree, variable_count)
    total = degree + variable_count
    support = 1
    for index in range(1, steps + 1):
        numerator = support * (total - steps + index)
        if numerator > MAX_POLYNOMIAL_TERMS * index:
            return MAX_POLYNOMIAL_TERMS + 1
        support = numerator // index
    return support


def _support_bound(candidate: int, degree: int, variables: frozenset[str]) -> int:
    return min(
        candidate,
        _bounded_monomial_support(degree, len(variables)),
    )


def _is_literal_zero_add(expression: PolynomialAdd) -> bool:
    if not all(
        isinstance(operand, PolynomialLiteral) for operand in expression.operands
    ):
        return False
    return (
        sum(
            (
                operand.value.as_fraction()
                for operand in expression.operands
                if isinstance(operand, PolynomialLiteral)
            ),
            Fraction(),
        )
        == 0
    )


def _addition_numerator_bits(
    metrics: list[_ExpressionMetrics], denominator: int | None
) -> int:
    if not metrics:
        return 1
    if denominator is None:
        return _MAX_EXPRESSION_COEFFICIENT_BITS + 1
    numerator_bits = 0
    for child in metrics:
        if child.denominator is None:
            return _MAX_EXPRESSION_COEFFICIENT_BITS + 1
        scale = denominator // child.denominator
        numerator_bits = max(
            numerator_bits,
            child.numerator_bits + _bit_growth(scale),
        )
    return min(
        _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
        numerator_bits + _ceil_log2(len(metrics)),
    )


def _decimal_digits_from_bits(bits: int) -> int:
    """Upper-bound decimal digits without converting a large integer."""

    if bits > _MAX_EXPRESSION_COEFFICIENT_BITS:
        return MAX_CANONICAL_RATIONAL_DIGITS + 1
    # 30103 / 100000 is just above log10(2), so this rounds conservatively.
    return (bits * 30_103 + 99_999) // 100_000 + 1


def _representation_digits(
    support: int, numerator_bits: int, denominator_bits: int
) -> int:
    component_digits = _decimal_digits_from_bits(
        numerator_bits
    ) + _decimal_digits_from_bits(denominator_bits)
    return _bounded_product(
        support,
        component_digits,
        _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS,
    )


def _digits_of_rational(value: Fraction) -> int:
    return _decimal_digits_from_bits(
        max(1, abs(value.numerator).bit_length())
    ) + _decimal_digits_from_bits(_denominator_bits(value.denominator))


def _scale_monomial(
    monomial: frozenset[tuple[str, int]] | None, exponent: int
) -> frozenset[tuple[str, int]] | None:
    if monomial is None:
        return None
    if exponent == 0:
        return frozenset()
    return frozenset(
        (name, power * exponent) for name, power in monomial if power * exponent
    )


def _multiply_monomials(
    children: list[_ExpressionMetrics],
) -> frozenset[tuple[str, int]] | None:
    if any(child.monomial is None for child in children):
        return None
    powers: dict[str, int] = {}
    for child in children:
        for name, power in child.monomial or ():
            powers[name] = powers.get(name, 0) + power
    return frozenset((name, power) for name, power in powers.items() if power)


def _addends_are_disjoint(children: list[_ExpressionMetrics]) -> bool:
    monomials = [child.monomial for child in children if not child.zero]
    return bool(monomials) and all(
        monomial is not None for monomial in monomials
    ) and len(set(monomials)) == len(monomials)


def _product_total_coefficient_digits(
    left_support: int,
    left_digits: int,
    right_support: int,
    right_digits: int,
) -> int:
    """Bound the aggregate coefficient digits of a sparse product.

    Distinct monomial products charge ``|g| * digits(f) + |f| * digits(g)``.
    Collisions can only reduce the number of terms, so this remains sound.
    """

    return min(
        _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS + 1,
        _bounded_product(left_support, right_digits, _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS)
        + _bounded_product(
            right_support, left_digits, _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS
        ),
    )


def _metrics(expression: PolynomialExpression) -> _ExpressionMetrics:
    denominator: int | None
    if isinstance(expression, PolynomialLiteral):
        require_bounded_rational(expression.value, max_digits=128, label="literal")
        numerator_bits = max(1, abs(expression.value.num).bit_length())
        return _ExpressionMetrics(
            nodes=1,
            support=1,
            expansion_terms=1,
            degree=0,
            variables=frozenset(),
            numerator_bits=numerator_bits,
            denominator=expression.value.den,
            zero=expression.value.num == 0,
            constant=expression.value.as_fraction(),
            monomial=frozenset(),
            total_coefficient_digits=_digits_of_rational(
                expression.value.as_fraction()
            ),
            maximum_numerator_bits=numerator_bits,
            maximum_denominator_bits=_denominator_bits(expression.value.den),
            work=1,
            intermediate_digits=_representation_digits(
                1, numerator_bits, _denominator_bits(expression.value.den)
            ),
        )
    if isinstance(expression, PolynomialVariableExpression):
        variables = frozenset((expression.name,))
        return _ExpressionMetrics(
            nodes=1,
            support=1,
            expansion_terms=1,
            degree=1,
            variables=variables,
            numerator_bits=1,
            denominator=1,
            zero=False,
            constant=None,
            monomial=frozenset(((expression.name, 1),)),
            total_coefficient_digits=2,
            maximum_numerator_bits=1,
            maximum_denominator_bits=0,
            work=1,
            intermediate_digits=_representation_digits(1, 1, 0),
        )
    if isinstance(expression, PolynomialPower):
        base = _metrics(expression.base)
        exponent = expression.exponent
        if exponent == 0:
            return _ExpressionMetrics(
                nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
                support=1,
                expansion_terms=1,
                degree=0,
                variables=frozenset(),
                numerator_bits=1,
                denominator=1,
                zero=False,
                constant=Fraction(1),
                monomial=frozenset(),
                total_coefficient_digits=2,
                maximum_numerator_bits=max(base.maximum_numerator_bits, 1),
                maximum_denominator_bits=base.maximum_denominator_bits,
                work=base.work,
                intermediate_digits=max(base.intermediate_digits, 2),
            )
        degree = min(MAX_POLYNOMIAL_EXPONENT + 1, base.degree * exponent)
        variables = base.variables
        support = _support_bound(
            _bounded_power(base.support, exponent, MAX_POLYNOMIAL_TERMS),
            degree,
            variables,
        )
        expansion_terms = _bounded_power(
            base.expansion_terms, exponent, MAX_POLYNOMIAL_TERMS
        )
        zero = base.zero
        constant: Fraction | None
        if zero:
            numerator_bits = 1
            denominator = 1
            constant = Fraction(0)
        elif base.constant is not None:
            projected_numerator_bits = min(
                _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
                max(1, base.numerator_bits) * exponent
                + _ceil_log2(max(1, base.support)) * exponent,
            )
            projected_denominator = _bounded_denominator_power(
                base.denominator, exponent
            )
            if (
                projected_numerator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
                or projected_denominator is None
                or _denominator_bits(projected_denominator)
                > _MAX_EXPRESSION_COEFFICIENT_BITS
            ):
                constant = None
                numerator_bits = projected_numerator_bits
                denominator = projected_denominator
            else:
                constant = base.constant**exponent
                numerator_bits = min(
                    _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
                    max(1, abs(constant.numerator).bit_length()),
                )
                denominator = constant.denominator
        else:
            constant = None
            numerator_bits = min(
                _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
                base.numerator_bits * exponent
                + _ceil_log2(max(1, base.support)) * exponent,
            )
            denominator = _bounded_denominator_power(base.denominator, exponent)
        work = base.work
        result_expansion_terms = 1
        base_expansion_terms = base.expansion_terms
        remaining = exponent
        while remaining:
            if remaining & 1:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(
                        result_expansion_terms,
                        base_expansion_terms,
                        _MAX_EXPRESSION_WORK,
                    ),
                )
                result_expansion_terms = _bounded_product(
                    result_expansion_terms,
                    base_expansion_terms,
                    MAX_POLYNOMIAL_TERMS,
                )
            remaining //= 2
            if remaining:
                work = min(
                    _MAX_EXPRESSION_WORK + 1,
                    work
                    + _bounded_product(
                        base_expansion_terms,
                        base_expansion_terms,
                        _MAX_EXPRESSION_WORK,
                    ),
                )
                base_expansion_terms = _bounded_product(
                    base_expansion_terms,
                    base_expansion_terms,
                    MAX_POLYNOMIAL_TERMS,
                )
        return _ExpressionMetrics(
            nodes=min(_MAX_EXPRESSION_NODES + 1, base.nodes + 1),
            support=support,
            expansion_terms=expansion_terms,
            degree=degree,
            variables=variables,
            numerator_bits=numerator_bits,
            denominator=denominator,
            zero=zero,
            constant=constant,
            monomial=_scale_monomial(base.monomial, exponent),
            total_coefficient_digits=(
                _digits_of_rational(constant)
                if constant is not None
                else _representation_digits(
                    support, numerator_bits, _denominator_bits(denominator)
                )
            ),
            maximum_numerator_bits=max(base.maximum_numerator_bits, numerator_bits),
            maximum_denominator_bits=max(
                base.maximum_denominator_bits,
                _denominator_bits(denominator),
            ),
            work=work,
            intermediate_digits=max(
                base.intermediate_digits,
                _representation_digits(
                    support, numerator_bits, _denominator_bits(denominator)
                ),
            ),
        )
    child_metrics = [_metrics(operand) for operand in expression.operands]
    nodes = min(_MAX_EXPRESSION_NODES + 1, 1 + sum(row.nodes for row in child_metrics))
    variables = frozenset().union(*(row.variables for row in child_metrics))
    if isinstance(expression, PolynomialAdd):
        expansion_terms = _bounded_sum(
            tuple(row.expansion_terms for row in child_metrics), MAX_POLYNOMIAL_TERMS
        )
        degree = max(row.degree for row in child_metrics)
        disjoint = _addends_are_disjoint(child_metrics)
        constant: Fraction | None
        all_constant = bool(child_metrics) and all(
            child.constant is not None for child in child_metrics
        )
        projected_denominator = (
            1
            if child_metrics and all(child.zero for child in child_metrics)
            else _bounded_denominator_lcm(child_metrics, skip_zero=True)
        )
        if all_constant and projected_denominator is not None:
            constant = sum(
                (child.constant for child in child_metrics),
                start=Fraction(),
            )
            zero = constant == 0
        elif all_constant:
            constant = None
            zero = False
        else:
            constant = None
            zero = all(child.zero for child in child_metrics) or _is_literal_zero_add(
                expression
            )
        active_metrics = [child for child in child_metrics if not child.zero]
        if zero:
            common_denominator = 1
            common_numerator_bits = 1
        elif disjoint:
            common_denominator = 1
            for child in child_metrics:
                if child.zero:
                    continue
                if child.denominator is None:
                    common_denominator = None
                    break
                if child.denominator > common_denominator:
                    common_denominator = child.denominator
            common_numerator_bits = max(
                (child.numerator_bits for child in child_metrics if not child.zero),
                default=1,
            )
        else:
            common_denominator = projected_denominator
            common_numerator_bits = _addition_numerator_bits(
                active_metrics,
                common_denominator,
            )
        raw_denominator = common_denominator
        denominator = 1 if zero else common_denominator
        numerator_bits = 1 if zero else common_numerator_bits
        support = _support_bound(
            _bounded_sum(
                tuple(row.support for row in child_metrics), MAX_POLYNOMIAL_TERMS
            ),
            degree,
            variables,
        )
        work = _bounded_sum(
            tuple(row.work + row.expansion_terms for row in child_metrics),
            _MAX_EXPRESSION_WORK,
        )
        carry_digits = _decimal_digits_from_bits(
            _ceil_log2(max(1, len(child_metrics)))
        )
        if zero:
            total_coefficient_digits = 1
        elif disjoint:
            total_coefficient_digits = min(
                _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS + 1,
                sum(child.total_coefficient_digits for child in child_metrics),
            )
        else:
            lcm_bits = _denominator_bits(common_denominator)
            scaled_digits = 0
            for child in active_metrics:
                if child.denominator is None or common_denominator is None:
                    scaled_digits = _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS + 1
                    break
                scale = common_denominator // child.denominator
                scaled_digits = min(
                    _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS + 1,
                    scaled_digits
                    + _representation_digits(
                        child.support,
                        child.numerator_bits + _bit_growth(scale),
                        lcm_bits,
                    ),
                )
            total_coefficient_digits = max(
                scaled_digits,
                min(
                    _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS + 1,
                    sum(child.total_coefficient_digits for child in child_metrics)
                    + _bounded_product(
                        support,
                        carry_digits,
                        _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS,
                    ),
                ),
            )
        maximum_numerator_bits = max(
            common_numerator_bits,
            *(child.maximum_numerator_bits for child in child_metrics),
        )
        if disjoint:
            maximum_denominator_bits = max(
                (child.maximum_denominator_bits for child in child_metrics),
                default=0,
            )
        else:
            maximum_denominator_bits = max(
                _denominator_bits(common_denominator),
                *(child.maximum_denominator_bits for child in child_metrics),
            )
    else:
        disjoint = False
        expansion_terms = 1
        support = 1
        work = 0
        current_expansion_terms = 1
        total_coefficient_digits = 1
        accumulated_support = 1
        for child in child_metrics:
            expansion_terms = _bounded_product(
                expansion_terms, child.expansion_terms, MAX_POLYNOMIAL_TERMS
            )
            total_coefficient_digits = _product_total_coefficient_digits(
                accumulated_support,
                total_coefficient_digits,
                child.support,
                child.total_coefficient_digits,
            )
            accumulated_support = _bounded_product(
                accumulated_support, child.support, MAX_POLYNOMIAL_TERMS
            )
            support = _bounded_product(support, child.support, MAX_POLYNOMIAL_TERMS)
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work + child.work,
            )
            work = min(
                _MAX_EXPRESSION_WORK + 1,
                work
                + _bounded_product(
                    current_expansion_terms,
                    child.expansion_terms,
                    _MAX_EXPRESSION_WORK,
                ),
            )
            current_expansion_terms = _bounded_product(
                current_expansion_terms,
                child.expansion_terms,
                MAX_POLYNOMIAL_TERMS,
            )
        degree = min(
            MAX_POLYNOMIAL_EXPONENT + 1,
            sum(row.degree for row in child_metrics),
        )
        support = _support_bound(support, degree, variables)
        common_numerator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            _bounded_sum(
                tuple(row.numerator_bits for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
            + _bounded_sum(
                tuple(_ceil_log2(max(1, row.support)) for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            ),
        )
        denominator = 1
        for child in child_metrics:
            denominator = _bounded_denominator_product(denominator, child.denominator)
            if denominator is None:
                break
        raw_denominator = denominator
        if any(child.zero for child in child_metrics):
            constant = Fraction(0)
        elif child_metrics and all(
            child.constant is not None for child in child_metrics
        ):
            constant = Fraction(1)
            for child in child_metrics:
                constant *= child.constant
        else:
            constant = None
        zero = constant == 0 if constant is not None else any(
            child.zero for child in child_metrics
        )
        numerator_bits = 1 if zero else common_numerator_bits
        if zero:
            denominator = 1
            total_coefficient_digits = 1
        maximum_numerator_bits = max(
            common_numerator_bits,
            *(child.maximum_numerator_bits for child in child_metrics),
        )
        maximum_denominator_bits = max(
            _denominator_bits(raw_denominator),
            *(child.maximum_denominator_bits for child in child_metrics),
        )
    intermediate_digits = max(
        (row.intermediate_digits for row in child_metrics),
        default=0,
    )
    intermediate_digits = max(intermediate_digits, total_coefficient_digits)
    return _ExpressionMetrics(
        nodes=nodes,
        support=support,
        expansion_terms=expansion_terms,
        degree=degree,
        variables=variables,
        numerator_bits=numerator_bits,
        denominator=denominator,
        zero=zero,
        constant=constant,
        monomial=(
            next((child.monomial for child in child_metrics if not child.zero), frozenset())
            if disjoint
            and sum(1 for child in child_metrics if not child.zero) == 1
            else _multiply_monomials(child_metrics)
            if not isinstance(expression, PolynomialAdd)
            else None
        ),
        total_coefficient_digits=total_coefficient_digits,
        maximum_numerator_bits=maximum_numerator_bits,
        maximum_denominator_bits=maximum_denominator_bits,
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
        or metrics.support > MAX_POLYNOMIAL_TERMS
        or metrics.degree > MAX_POLYNOMIAL_EXPONENT
        or metrics.numerator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
        or _denominator_bits(metrics.denominator) > _MAX_EXPRESSION_COEFFICIENT_BITS
        or metrics.maximum_numerator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
        or metrics.maximum_denominator_bits > _MAX_EXPRESSION_COEFFICIENT_BITS
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
