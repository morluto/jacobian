"""Non-evaluating typed polynomial expression normalization."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, ValidationError, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
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
        """Reject deep or oversized raw trees before copying or parsing them."""

        if isinstance(value, Mapping):
            _bound_raw_request(value)
        return canonicalize_json_containers(value)

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
_OWNER_DEADLINE_SECONDS = 60.0
_CHECKPOINT_STRIDE = 256
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
    denominator_mass_bits: int
    work: int
    intermediate_digits: int
    support_keys: frozenset[tuple[tuple[str, int], ...]] | None = None
    termwise_disjoint: bool = False
    single_term: tuple[frozenset[tuple[str, int]], Fraction] | None = None


class _MalformedExpressionError(ValueError):
    """A node violates the closed expression grammar or operand shape."""


def _is_expression_node(node: object) -> bool:
    """Return whether ``node`` is a raw mapping or a recognized AST node."""

    return isinstance(
        node,
        (
            Mapping,
            PolynomialLiteral,
            PolynomialVariableExpression,
            PolynomialAdd,
            PolynomialMultiply,
            PolynomialPower,
        ),
    )


def _bounded_operands(operands: object) -> tuple[object, ...]:
    """Materialize operands under the arity cap and require expression nodes."""

    if not isinstance(operands, (list, tuple)):
        raise _MalformedExpressionError(
            "expression operands must be a bounded sequence"
        )
    bounded: list[object] = []
    for operand in operands:
        if len(bounded) >= 64:
            raise _MalformedExpressionError(
                "expression nodes may have at most 64 operands"
            )
        if not _is_expression_node(operand):
            raise _MalformedExpressionError(
                "every expression operand must be a mapping or recognized node"
            )
        bounded.append(operand)
    return tuple(bounded)


def _expression_children(node: object) -> tuple[object, ...]:
    if isinstance(node, (PolynomialAdd, PolynomialMultiply)):
        operands = getattr(node, "operands", ())
        return _bounded_operands(operands)
    if isinstance(node, PolynomialPower):
        base = getattr(node, "base", None)
        return (base,) if base is not None else ()
    if (
        isinstance(node, (PolynomialLiteral, PolynomialVariableExpression))
        or node is None
    ):
        return ()
    if isinstance(node, Mapping):
        kind = node.get("kind")
        if kind in ("ADD", "MULTIPLY"):
            return _bounded_operands(node.get("operands"))
        if kind == "POWER":
            if "base" not in node:
                raise _MalformedExpressionError("a POWER node requires a base")
            return (node["base"],)
        if kind is None:
            return ()
        if kind in ("VARIABLE", "LITERAL"):
            return ()
        raise _MalformedExpressionError(f"unrecognized expression node kind: {kind!r}")
    raise _MalformedExpressionError(
        f"unrecognized expression node: {type(node).__name__}"
    )


def _require_bounded_variables(variables: list[object] | tuple[object, ...]) -> None:
    """Require scalar string axis entries before the recursive copy touches them."""

    for variable in variables:
        if not isinstance(variable, str):
            raise ValueError("variables must be scalar strings")


def _bound_raw_request(value: Mapping[str, object]) -> None:
    """Bound every raw request field before the recursive canonicalization copy."""

    allowed = {"coefficient_domain", "variables", "expression"}
    # Iterate keys instead of materializing a set, and reject the first
    # unexpected key or any surplus key, so a request with millions of extra
    # fields never allocates a sorted copy of them.
    for index, key in enumerate(value):
        if index >= len(allowed):
            raise ValueError("expression requests may not carry unexpected fields")
        if key not in allowed:
            raise ValueError(
                f"expression requests may not carry the unexpected field {key!r}"
            )
    variables = value.get("variables")
    if variables is not None:
        if not isinstance(variables, (list, tuple)):
            raise ValueError("variables must be a bounded sequence")
        if len(variables) > MAX_POLYNOMIAL_VARIABLES:
            raise ValueError("variables exceed the admitted axis bound")
        _require_bounded_variables(variables)
    domain = value.get("coefficient_domain")
    if domain is not None and not isinstance(domain, str):
        raise ValueError("coefficient_domain must be a string")
    _bound_raw_expression(value.get("expression"))


def _bound_raw_expression(expression: object) -> None:
    """Bound AST depth and cardinality for mappings and validated models.

    Malformed recognized nodes are rejected here rather than treated as
    childless, so an unexpected container field cannot reach the recursive
    canonicalization copy without first being bounded.
    """

    stack: list[tuple[object, int, tuple[int, ...]]] = [(expression, 1, ())]
    count = 0
    while stack:
        node, depth, path = stack.pop()
        identity = id(node)
        if identity in path:
            raise ValueError("expression nodes may not form a cycle")
        count += 1
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ValueError(
                f"expression depth exceeds the {_MAX_EXPRESSION_DEPTH}-node path bound"
            )
        if count > _MAX_EXPRESSION_NODES:
            raise ValueError(f"expression node count exceeds {_MAX_EXPRESSION_NODES}")
        children = _expression_children(node)
        if len(children) > 64:
            raise ValueError("expression nodes may have at most 64 operands")
        if isinstance(node, Mapping):
            _require_bounded_mapping_fields(node)
        child_path = (*path, identity)
        stack.extend((child, depth + 1, child_path) for child in children)


def _require_bounded_mapping_fields(node: Mapping[str, object]) -> None:
    """Reject unexpected keys and unbounded scalar/container fields."""

    kind = node.get("kind")
    allowed = {
        "ADD": {"kind", "operands"},
        "MULTIPLY": {"kind", "operands"},
        "POWER": {"kind", "base", "exponent"},
        "VARIABLE": {"kind", "name"},
        "LITERAL": {"kind", "value"},
    }.get(kind if isinstance(kind, str) else "")
    if allowed is None:
        raise _MalformedExpressionError(f"unrecognized expression node kind: {kind!r}")
    unexpected = set(node).difference(allowed)
    if unexpected:
        raise _MalformedExpressionError(
            "expression nodes may not carry unexpected fields: "
            + ", ".join(sorted(map(str, unexpected)))
        )
    value = node.get("value")
    if "value" in node:
        if isinstance(value, (list, tuple)):
            raise _MalformedExpressionError(
                "LITERAL value must be a num/den object, not a sequence"
            )
        if isinstance(value, Mapping) and (
            set(value).difference({"num", "den"}) or len(value) > 2
        ):
            raise _MalformedExpressionError(
                "LITERAL value must contain only num and den"
            )
        if isinstance(value, Mapping):
            for component in value.values():
                if isinstance(component, (list, tuple, Mapping)):
                    raise _MalformedExpressionError(
                        "LITERAL components must be scalars, not containers"
                    )
    for scalar_field in ("name", "exponent"):
        if scalar_field in node and isinstance(
            node[scalar_field], (list, tuple, Mapping)
        ):
            raise _MalformedExpressionError(
                f"{scalar_field} must be a scalar, not a container"
            )


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


def _admit_literal(value: CanonicalRational) -> None:
    try:
        require_bounded_rational(value, max_digits=128, label="literal")
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.literal_bound",
            message="expression literals exceed the admitted 128-digit source bound",
        ) from error


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


def _bounded_single_term(
    monomial: frozenset[tuple[str, int]] | None,
    coefficient: Fraction,
) -> tuple[frozenset[tuple[str, int]], Fraction] | None:
    """Return an exact single-monomial state when both parts stay representable."""

    if monomial is None:
        return None
    if (
        abs(coefficient.numerator).bit_length() > _MAX_EXPRESSION_COEFFICIENT_BITS
        or coefficient.denominator.bit_length() > _MAX_EXPRESSION_COEFFICIENT_BITS
    ):
        return None
    return (monomial, coefficient)


def _single_term_power(
    base: tuple[frozenset[tuple[str, int]], Fraction] | None,
    exponent: int,
) -> tuple[frozenset[tuple[str, int]], Fraction] | None:
    """Raise an exact single term to a power without over-height expansion."""

    if base is None or exponent < 0:
        return None
    monomial, coefficient = base
    if coefficient == 0:
        return _bounded_single_term(frozenset(), Fraction(0))
    bound = _MAX_EXPRESSION_COEFFICIENT_BITS
    if (
        abs(coefficient.numerator).bit_length() * exponent > bound
        or coefficient.denominator.bit_length() * exponent > bound
    ):
        return None
    return _bounded_single_term(
        _scale_monomial(monomial, exponent), coefficient**exponent
    )


def _multiply_single_terms(
    children: list[_ExpressionMetrics],
) -> tuple[frozenset[tuple[str, int]], Fraction] | None:
    """Multiply exact single-term children into one term when representable."""

    powers: dict[str, int] = {}
    coefficient = Fraction(1)
    for child in children:
        if child.single_term is None:
            return None
        monomial, child_coefficient = child.single_term
        # Check the product's size before materializing it, so an over-envelope
        # constant product is never built.
        if (
            abs(coefficient.numerator).bit_length()
            + abs(child_coefficient.numerator).bit_length()
            > _MAX_EXPRESSION_COEFFICIENT_BITS
            or coefficient.denominator.bit_length()
            + child_coefficient.denominator.bit_length()
            > _MAX_EXPRESSION_COEFFICIENT_BITS
        ):
            return None
        coefficient = coefficient * child_coefficient
        for name, power in monomial:
            powers[name] = powers.get(name, 0) + power
    return _bounded_single_term(
        frozenset((name, power) for name, power in powers.items() if power),
        coefficient,
    )


def _exact_addend_cancellation(
    child_metrics: list[_ExpressionMetrics],
) -> dict[frozenset[tuple[str, int]], Fraction] | None:
    """Sum exact single-term addends by monomial, or ``None`` if not all are single.

    Returns the per-monomial coefficient map when every nonzero addend is an
    exact single term. A monomial whose summed coefficient is zero has cancelled
    symbolically, which the sign-blind height aggregation cannot see.
    """

    groups: dict[frozenset[tuple[str, int]], Fraction] = {}
    for child in child_metrics:
        if child.zero:
            continue
        if child.single_term is None:
            return None
        monomial, coefficient = child.single_term
        existing = groups.get(monomial, Fraction())
        # Bound each running addition before it is materialized. For an exact
        # cancellation the denominators are equal, so the sum-of-bit-lengths
        # estimate still fits and the symbolic zero is found.
        if (
            existing.denominator.bit_length() + coefficient.denominator.bit_length()
            > _MAX_EXPRESSION_COEFFICIENT_BITS
            or abs(existing.numerator).bit_length()
            + abs(coefficient.numerator).bit_length()
            > _MAX_EXPRESSION_COEFFICIENT_BITS
        ):
            return None
        groups[monomial] = existing + coefficient
    return groups


def _support_keys_are_uniquely_decomposable(
    keys: frozenset[tuple[tuple[str, int], ...]] | None,
) -> bool:
    """Decide whether monomial products uniquely determine the factor multiset.

    A power's coefficients each combine only one factor multiset when distinct
    multisets of the same size cannot multiply to the same monomial. That holds
    exactly when the support's *homogenized* exponent vectors ``(v_i, 1)`` are
    linearly independent over the rationals: a dependence among them is an
    affine relation ``sum c_i v_i = 0`` with ``sum c_i = 0``, whose
    nonnegative and nonpositive parts are distinct equal-size multisets with an
    equal sum.
    """

    if not keys:
        return False
    vectors = [dict(key) for key in keys]
    if len(vectors) <= 1:
        return True
    names = sorted({name for vector in vectors for name in vector})
    # A homogenized coordinate is appended to every vector, so more vectors
    # than ambient coordinates plus one cannot be independent.
    if len(vectors) > len(names) + 1:
        return False
    # Gaussian elimination over Fractions on the homogenized vectors.
    rows = [
        [Fraction(vector.get(name, 0)) for name in names] + [Fraction(1)]
        for vector in vectors
    ]
    rank = 0
    column_count = len(names) + 1
    for column in range(column_count):
        pivot = next(
            (row for row in range(rank, len(rows)) if rows[row][column] != 0), None
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row != rank and rows[row][column] != 0:
                factor = rows[row][column]
                rows[row] = [
                    value - factor * base
                    for value, base in zip(rows[row], rows[rank], strict=True)
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank == len(vectors)


def _power_support_keys(
    keys: frozenset[tuple[tuple[str, int], ...]] | None, exponent: int
) -> frozenset[tuple[tuple[str, int], ...]] | None:
    """Return the complete support keys of a power, or ``None`` when unknown.

    Scaling each base monomial records only the pure powers; the true support
    of ``(sum m_i)^k`` is the ``k``-fold Minkowski sum, which includes mixed
    products. Only a single base monomial (or the identity power) has a known
    complete support here.
    """

    if keys is None:
        return None
    if exponent <= 1:
        return keys
    if len(keys) != 1:
        return None
    scaled: set[tuple[tuple[str, int], ...]] = set()
    for key in keys:
        scaled.add(
            tuple(sorted((name, power * exponent) for name, power in key if power))
        )
    return frozenset(scaled)


def _multiply_support_keys(
    left: frozenset[tuple[tuple[str, int], ...]] | None,
    right: frozenset[tuple[tuple[str, int], ...]] | None,
) -> frozenset[tuple[tuple[str, int], ...]] | None:
    if left is None or right is None:
        return None
    if len(left) * len(right) > MAX_POLYNOMIAL_TERMS:
        return None
    product: set[tuple[tuple[str, int], ...]] = set()
    for left_key in left:
        left_map = dict(left_key)
        for right_key in right:
            merged = dict(left_map)
            for name, power in right_key:
                merged[name] = merged.get(name, 0) + power
            product.add(
                tuple(sorted((name, power) for name, power in merged.items() if power))
            )
    return frozenset(product)


def _product_support_collision(
    child_metrics: list[_ExpressionMetrics],
) -> bool:
    """Decide whether two choices of factor monomials can share an exponent.

    When every factor's tracked support multiplies to exactly the product of
    its sizes, each output monomial selects one term per factor, so no two
    factor choices are ever summed and each factor contributes its own reduced
    denominator. Any other shape - an untracked support, a product above the
    admitted term ceiling, or a collapsed key set - can collide, and the
    caller must then bound the numerator as a sum over a common denominator.
    """

    combined: frozenset[tuple[tuple[str, int], ...]] = frozenset(((),))
    expected = 1
    for child in child_metrics:
        if child.support_keys is None:
            return True
        expected *= len(child.support_keys)
        if expected > MAX_POLYNOMIAL_TERMS:
            return True
        multiplied = _multiply_support_keys(combined, child.support_keys)
        if multiplied is None:
            return True
        combined = multiplied
    return len(combined) != expected


def _product_denominator_scaling_bits(
    child_metrics: list[_ExpressionMetrics],
    *,
    colliding: bool,
) -> int:
    """Bound the numerator growth from writing colliding products over one denominator.

    A colliding output coefficient sums products ``n_1/d_1 * ... * n_k/d_k``, so
    each summand is scaled by the denominators the other factors contributed.
    The combined denominator divides ``d_1 * ... * d_k``, so summand ``i``
    grows by at most the bits of every other factor denominator: ``(k - 1)``
    times the total. Each factor contributes its whole denominator mass, not
    only its representative denominator, because a colliding coefficient can
    combine every denominator the factor hides. Without collisions this scaling
    never happens and the bound is zero. This mirrors
    ``_addition_numerator_bits``, which already charges an addition for reaching
    its common denominator.
    """

    if not colliding:
        return 0
    total_bits = 0
    for child in child_metrics:
        if child.denominator is None:
            return _MAX_EXPRESSION_COEFFICIENT_BITS + 1
        total_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            total_bits
            + max(
                _denominator_bits(child.denominator),
                child.denominator_mass_bits,
            ),
        )
    return min(
        _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
        total_bits * max(0, len(child_metrics) - 1),
    )


def _addends_are_disjoint(children: list[_ExpressionMetrics]) -> bool:
    known = [child.support_keys for child in children if not child.zero]
    concrete = [keys for keys in known if keys is not None]
    if known and len(concrete) == len(known):
        seen: set[tuple[tuple[str, int], ...]] = set()
        for keys in concrete:
            for key in keys:
                if key in seen:
                    return False
                seen.add(key)
        return bool(seen)
    monomials = [child.monomial for child in children if not child.zero]
    return (
        bool(monomials)
        and all(monomial is not None for monomial in monomials)
        and len(set(monomials)) == len(monomials)
    )


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
        _bounded_product(
            left_support, right_digits, _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS
        )
        + _bounded_product(
            right_support, left_digits, _MAX_EXPRESSION_TOTAL_COEFFICIENT_DIGITS
        ),
    )


def _metrics(
    expression: PolynomialExpression,
    multiply_orders: dict[int, tuple[int, ...]] | None = None,
) -> _ExpressionMetrics:
    denominator: int | None
    if isinstance(expression, PolynomialLiteral):
        _admit_literal(expression.value)
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
            denominator_mass_bits=_denominator_bits(expression.value.den),
            work=1,
            intermediate_digits=_representation_digits(
                1, numerator_bits, _denominator_bits(expression.value.den)
            ),
            support_keys=frozenset() if expression.value.num == 0 else frozenset(((),)),
            termwise_disjoint=True,
            single_term=(frozenset(), expression.value.as_fraction()),
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
            denominator_mass_bits=0,
            work=1,
            intermediate_digits=_representation_digits(1, 1, 0),
            support_keys=frozenset((((expression.name, 1),),)),
            termwise_disjoint=True,
            single_term=(
                frozenset(((expression.name, 1),)),
                Fraction(1),
            ),
        )
    if isinstance(expression, PolynomialPower):
        base = _metrics(expression.base, multiply_orders)
        exponent = expression.exponent
        if exponent == 0:
            # A zero power is the constant one and never expands the base, so
            # only its node count is charged; coefficient, work, and
            # representation growth all belong to the discarded base.
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
                maximum_numerator_bits=1,
                maximum_denominator_bits=0,
                denominator_mass_bits=0,
                work=1,
                intermediate_digits=2,
                support_keys=frozenset(((),)),
                termwise_disjoint=True,
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
        if base.termwise_disjoint and _support_keys_are_uniquely_decomposable(
            base.support_keys
        ):
            powered_denominator_bits = min(
                _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
                max(base.maximum_denominator_bits, 1) * exponent
                if base.maximum_denominator_bits
                else 0,
            )
        else:
            powered_denominator_bits = min(
                _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
                base.denominator_mass_bits * exponent,
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
                else base.total_coefficient_digits
                if exponent == 1
                else _representation_digits(
                    support, numerator_bits, _denominator_bits(denominator)
                )
            ),
            maximum_numerator_bits=max(base.maximum_numerator_bits, numerator_bits),
            maximum_denominator_bits=max(
                base.maximum_denominator_bits,
                _denominator_bits(denominator),
                powered_denominator_bits,
            ),
            denominator_mass_bits=powered_denominator_bits,
            work=work,
            intermediate_digits=(
                base.intermediate_digits
                if exponent == 1
                else max(
                    base.intermediate_digits,
                    _representation_digits(
                        support, numerator_bits, _denominator_bits(denominator)
                    ),
                )
            ),
            support_keys=_power_support_keys(base.support_keys, exponent),
            termwise_disjoint=base.termwise_disjoint
            and (
                exponent <= 1
                or (base.support_keys is not None and len(base.support_keys) <= 1)
            ),
            single_term=_single_term_power(base.single_term, exponent),
        )
    if isinstance(expression, (PolynomialAdd, PolynomialMultiply)):
        return _nary_expression_metrics(expression, multiply_orders)
    raise TypeError("expression kind is not a polynomial operator")


def _nary_expression_metrics(  # noqa: C901
    expression: PolynomialAdd | PolynomialMultiply,
    multiply_orders: dict[int, tuple[int, ...]] | None = None,
) -> _ExpressionMetrics:
    indexed_children = [
        (index, _metrics(operand, multiply_orders))
        for index, operand in enumerate(expression.operands)
    ]
    if isinstance(expression, PolynomialMultiply):
        # Multiplication is commutative. Grow the running product from the
        # smallest admitted expansion bound so sparse factors do not inflate
        # every subsequent pairwise convolution. The same stable plan drives
        # admission and evaluation.
        indexed_children.sort(key=lambda row: (row[1].expansion_terms, row[0]))
        if multiply_orders is not None:
            multiply_orders[id(expression)] = tuple(
                index for index, _ in indexed_children
            )
    child_metrics = [metrics for _, metrics in indexed_children]
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
        # Exact single-monomial addends can cancel symbolically (for example
        # ``x/p - x/p``), which the sign-blind height aggregation cannot see and
        # would otherwise charge as one common denominator. All-constant sums
        # already reduce exactly in the constant branch below, so only
        # non-constant addends need this path.
        exact_addends = (
            None if all_constant else _exact_addend_cancellation(child_metrics)
        )
        exact_zero = exact_addends is not None and not any(exact_addends.values())
        survivors = (
            {
                monomial: coefficient
                for monomial, coefficient in exact_addends.items()
                if coefficient
            }
            if exact_addends is not None
            else {}
        )
        if all_constant and projected_denominator is not None:
            constant = sum(
                (
                    child.constant
                    for child in child_metrics
                    if child.constant is not None
                ),
                start=Fraction(),
            )
            zero = constant == 0
        elif all_constant:
            constant = None
            zero = False
        else:
            constant = None
            zero = (
                all(child.zero for child in child_metrics)
                or _is_literal_zero_add(expression)
                or exact_zero
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
        # ``_add`` copies the whole accumulated coefficient dictionary before
        # merging each operand, so an ADD must charge one running-support copy
        # per operand rather than only each operand's own support.
        accumulated_terms = 0
        accumulated_degree = 0
        accumulated_support = 0
        clone_work = 0
        for child in child_metrics:
            clone_work += accumulated_support
            accumulated_terms = min(
                MAX_POLYNOMIAL_TERMS, accumulated_terms + child.support
            )
            accumulated_degree = max(accumulated_degree, child.degree)
            accumulated_support = _support_bound(
                accumulated_terms, accumulated_degree, variables
            )
        work = _bounded_sum((work, clone_work), _MAX_EXPRESSION_WORK)
        carry_digits = _decimal_digits_from_bits(_ceil_log2(max(1, len(child_metrics))))
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
            denominator_mass_bits = _bounded_sum(
                tuple(child.denominator_mass_bits for child in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
        else:
            extras = tuple(
                max(0, child.denominator_mass_bits - child.maximum_denominator_bits)
                for child in child_metrics
            )
            hidden_bits = _bounded_sum(extras, _MAX_EXPRESSION_COEFFICIENT_BITS)
            maximum_denominator_bits = max(
                _denominator_bits(common_denominator),
                hidden_bits,
                *(child.maximum_denominator_bits for child in child_metrics),
            )
            denominator_mass_bits = max(
                maximum_denominator_bits,
                _denominator_bits(common_denominator),
            )
    else:
        disjoint = False
        expansion_terms = 1
        support = 1
        work = 0
        current_expansion_terms = 1
        total_coefficient_digits = 1
        accumulated_support = 1
        accumulated_degree = 0
        accumulated_variables: frozenset[str] = frozenset()
        for child in child_metrics:
            expansion_terms = _bounded_product(
                expansion_terms, child.expansion_terms, MAX_POLYNOMIAL_TERMS
            )
            # Charge the *attainable* support of each partial product, not the
            # raw Cartesian path count: a repeated factor such as
            # ``(A + A*x)**12`` has 4,096 expansion paths but only 13
            # monomials, and charging the raw count saturates the aggregate
            # digit envelope and rejects a cheap result.
            accumulated_degree = min(
                MAX_POLYNOMIAL_EXPONENT + 1, accumulated_degree + child.degree
            )
            accumulated_variables = accumulated_variables | child.variables
            next_support = _support_bound(
                _bounded_product(
                    accumulated_support, child.support, MAX_POLYNOMIAL_TERMS
                ),
                accumulated_degree,
                accumulated_variables,
            )
            total_coefficient_digits = _product_total_coefficient_digits(
                accumulated_support,
                total_coefficient_digits,
                min(child.support, next_support),
                child.total_coefficient_digits,
            )
            accumulated_support = next_support
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
        collides = _product_support_collision(child_metrics)
        common_numerator_bits = min(
            _MAX_EXPRESSION_COEFFICIENT_BITS + 1,
            _bounded_sum(
                tuple(row.numerator_bits for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
            + _bounded_sum(
                tuple(_ceil_log2(max(1, row.support)) for row in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
            + _product_denominator_scaling_bits(child_metrics, colliding=collides),
        )
        denominator = 1
        for child in child_metrics:
            denominator = _bounded_denominator_product(denominator, child.denominator)
            if denominator is None:
                break
        raw_denominator = denominator
        if any(child.zero for child in child_metrics):
            constant = Fraction(0)
        elif (
            child_metrics
            and denominator is not None
            and common_numerator_bits <= _MAX_EXPRESSION_COEFFICIENT_BITS
        ):
            product = Fraction(1)
            for child in child_metrics:
                factor = child.constant
                if factor is None:
                    constant = None
                    break
                product *= factor
            else:
                constant = product
        else:
            constant = None
        zero = (
            constant == 0
            if constant is not None
            else any(child.zero for child in child_metrics)
        )
        numerator_bits = 1 if zero else common_numerator_bits
        if zero:
            denominator = 1
            total_coefficient_digits = 1
        maximum_numerator_bits = max(
            common_numerator_bits,
            *(child.maximum_numerator_bits for child in child_metrics),
        )
        # When the combined product support has exactly the product of the
        # factor support sizes, no two factor choices collide, so each output
        # monomial selects exactly one denominator from each factor and
        # mutually exclusive denominators are not summed.
        if not collides:
            denominator_mass_bits = max(
                (child.denominator_mass_bits for child in child_metrics),
                default=0,
            )
        else:
            denominator_mass_bits = _bounded_sum(
                tuple(child.denominator_mass_bits for child in child_metrics),
                _MAX_EXPRESSION_COEFFICIENT_BITS,
            )
        maximum_denominator_bits = max(
            _denominator_bits(raw_denominator),
            denominator_mass_bits,
            *(child.maximum_denominator_bits for child in child_metrics),
        )
    intermediate_digits = max(
        (row.intermediate_digits for row in child_metrics),
        default=0,
    )
    intermediate_digits = max(intermediate_digits, total_coefficient_digits)
    combined_keys: frozenset[tuple[tuple[str, int], ...]] | None
    if isinstance(expression, PolynomialAdd):
        combined_keys = frozenset()
        for child in child_metrics:
            if child.zero:
                continue
            if child.support_keys is None:
                combined_keys = None
                break
            combined_keys = combined_keys | child.support_keys
        add_disjoint = disjoint
    else:
        add_disjoint = False
        combined_keys = frozenset(((),))
        for child in child_metrics:
            combined_keys = _multiply_support_keys(combined_keys, child.support_keys)
            if combined_keys is None:
                break
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
            next(
                (child.monomial for child in child_metrics if not child.zero),
                frozenset(),
            )
            if disjoint and sum(1 for child in child_metrics if not child.zero) == 1
            else _multiply_monomials(child_metrics)
            if not isinstance(expression, PolynomialAdd)
            else None
        ),
        total_coefficient_digits=total_coefficient_digits,
        maximum_numerator_bits=maximum_numerator_bits,
        maximum_denominator_bits=maximum_denominator_bits,
        denominator_mass_bits=denominator_mass_bits,
        work=work,
        intermediate_digits=intermediate_digits,
        support_keys=combined_keys,
        termwise_disjoint=add_disjoint
        or (
            not isinstance(expression, PolynomialAdd)
            and all(child.termwise_disjoint for child in child_metrics)
            and combined_keys is not None
        ),
        single_term=(
            (next(iter(survivors.items())) if len(survivors) == 1 else None)
            if isinstance(expression, PolynomialAdd)
            else _multiply_single_terms(child_metrics)
        ),
    )


def _bind_expansion_deadline() -> None:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + _OWNER_DEADLINE_SECONDS
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before polynomial expression expansion")


def _bound_source_expression(expression: object) -> None:
    """Bound an expression node tree, mapping failures to typed errors."""

    try:
        _bound_raw_expression(expression)
    except _MalformedExpressionError as exc:
        raise OperationDomainValidationError(
            location=("expression",),
            code="polynomial.expression.invalid_source",
            message=str(exc),
        ) from exc
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=("expression",),
            code="polynomial.expression.expansion_bound",
            message=str(exc),
        ) from exc


def _revalidate_expression_source(
    source: PolynomialExpressionSource,
) -> PolynomialExpressionSource:
    """Reject forged native AST nodes before expansion metrics."""

    if not isinstance(source, PolynomialExpressionSource):
        raise OperationDomainValidationError(
            location=("source",),
            code="polynomial.expression.invalid_source",
            message="expression source must be a PolynomialExpressionSource",
        )
    _bound_source_expression(getattr(source, "expression", None))
    try:
        return PolynomialExpressionSource.model_validate(
            {
                "coefficient_domain": getattr(source, "coefficient_domain", None),
                "variables": getattr(source, "variables", None),
                "expression": getattr(source, "expression", None),
            }
        )
    except ValidationError as exc:
        details = exc.errors()
        text = " ".join(str(item.get("msg", "")) for item in details)
        if "depth" in text or "node count" in text:
            raise OperationResourceAdmissionError(
                location=("expression",),
                code="polynomial.expression.expansion_bound",
                message=text or "expression expansion exceeds the admitted bound",
            ) from exc
        raise OperationDomainValidationError(
            location=("expression",),
            code="polynomial.expression.invalid_source",
            message="expression nodes must satisfy the closed grammar before expansion",
        ) from exc


def _admit_source_domain_claims(source: PolynomialExpressionSource) -> None:
    """Reject ZZ fractional literals and undeclared variables before expansion."""

    declared = set(source.variables)
    stack: list[PolynomialExpression] = [source.expression]
    while stack:
        node = stack.pop()
        if isinstance(node, PolynomialLiteral):
            denominator = getattr(getattr(node, "value", None), "den", None)
            if denominator is None:
                raise _invalid_expression_source(
                    "LITERAL nodes require a num/den value"
                )
            if source.coefficient_domain == "ZZ" and denominator != 1:
                raise OperationDomainValidationError(
                    location=("expression",),
                    code="polynomial.expression.nonintegral_literal",
                    message="ZZ expressions require integral literals",
                )
        elif isinstance(node, PolynomialVariableExpression):
            if getattr(node, "name", None) not in declared:
                raise OperationDomainValidationError(
                    location=("expression",),
                    code="polynomial.expression.undeclared_variable",
                    message="every expression variable must belong to the declared axis",
                )
        elif isinstance(node, (PolynomialAdd, PolynomialMultiply)):
            operands = getattr(node, "operands", None)
            if not isinstance(operands, (list, tuple)):
                raise _invalid_expression_source(
                    "ADD and MULTIPLY nodes require an operands sequence"
                )
            if not 1 <= len(operands) <= 64:
                raise _invalid_expression_source(
                    "ADD and MULTIPLY nodes require 1 to 64 operands"
                )
            stack.extend(operands)
        elif isinstance(node, PolynomialPower):
            base = getattr(node, "base", None)
            if base is None:
                raise _invalid_expression_source("POWER nodes require a base")
            exponent = getattr(node, "exponent", None)
            if isinstance(exponent, bool) or not isinstance(exponent, int):
                raise _invalid_expression_source(
                    "POWER nodes require an integer exponent"
                )
            if not 0 <= exponent <= 32:
                raise _invalid_expression_source(
                    "POWER nodes require an exponent between 0 and 32"
                )
            stack.append(base)


def _invalid_expression_source(message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("expression",),
        code="polynomial.expression.invalid_source",
        message=message,
    )


def _add(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result = dict(left)
    for index, (exponent, coefficient) in enumerate(right.items()):
        if index % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during polynomial expression addition")
        result[exponent] = result.get(exponent, Fraction()) + coefficient
        if not result[exponent]:
            del result[exponent]
    return result


def _multiply(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    products = 0
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            if products % _CHECKPOINT_STRIDE == 0:
                request_checkpoint("during polynomial expression multiplication")
            products += 1
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction()) + left_coefficient * right_coefficient
            )
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def normalize_polynomial_expression(  # noqa: C901
    source: PolynomialExpressionSource,
) -> PolynomialExpressionNormalizeResult:
    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return normalize_polynomial_expression(source)
    source = _revalidate_expression_source(source)
    _admit_source_domain_claims(source)
    _bound_source_expression(source.expression)
    multiply_orders: dict[int, tuple[int, ...]] = {}
    metrics = _metrics(source.expression, multiply_orders)
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
    _bind_expansion_deadline()
    variable_index = {
        variable: index for index, variable in enumerate(source.variables)
    }
    zero_exp = (0,) * len(source.variables)

    def evaluate(expression: PolynomialExpression) -> dict[tuple[int, ...], Fraction]:
        request_checkpoint("during polynomial expression expansion")
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
            order = multiply_orders[id(expression)]
            for index in order:
                result = _multiply(result, evaluate(expression.operands[index]))
            return result
        if expression.exponent == 0:
            # A zero power is the constant one; do not expand the base, which
            # admission cannot bound through this branch.
            return {zero_exp: Fraction(1)}
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
    terms: list[RationalPolynomialTerm] = []
    for index, (exponent, coefficient) in enumerate(
        sorted(coefficients.items(), reverse=True)
    ):
        if index % _CHECKPOINT_STRIDE == 0:
            request_checkpoint("during polynomial expression result construction")
        terms.append(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                exponents=exponent,
            )
        )
    polynomial = RationalPolynomial(
        variables=source.variables,
        polynomial=SparseRationalPolynomial(terms=tuple(terms)),
    )
    request_checkpoint("after polynomial expression result construction")
    return PolynomialExpressionNormalizeResult(source=source, polynomial=polynomial)


__all__ = ["normalize_polynomial_expression"]
