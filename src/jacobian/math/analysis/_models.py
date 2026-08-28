"""Typed contracts for rigorous real-function enclosures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.analysis.intervals import ClosedRationalInterval


def _validation_error(message: str) -> PydanticCustomError:
    """Project analysis-model invariants through a stable owner code."""

    return PydanticCustomError("analysis.invariant", message)


MAX_RATIONAL_DIGITS = 128
MAX_EXPRESSION_DEPTH = 16
MAX_EXPRESSION_NODES = 64
MAX_INTEGER_EXPONENT = 64
MAX_BOX_VARIABLES = 8
# Cap every retained exact preflight numerator and denominator at twice the
# maximum Arb work precision.  One Fraction binary operation or comparison can
# transiently combine two such components, so its temporary size is separately
# bounded by the derived double-width ceiling below.
MAX_BOX_INTERMEDIATE_BITS = 8_192
MAX_BOX_PREFLIGHT_TEMPORARY_BITS = 2 * MAX_BOX_INTERMEDIATE_BITS + 1
MAX_DYADIC_EXPONENT = 2**53 - 1
MAX_DYADIC_MANTISSA_DIGITS = 1_235

IntervalVariable = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]

type IntervalExpressionOp = Literal[
    "const",
    "var",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
]


def _bound_raw_rational(value: object, label: str) -> None:
    """Reject oversized components before canonical-rational construction."""

    if isinstance(value, CanonicalRational):
        components: tuple[object, object] = (value.num, value.den)
    elif isinstance(value, Mapping):
        components = (value.get("num"), value.get("den"))
    else:
        return
    if any(
        isinstance(component, str)
        and len(component) - component.startswith("-") > MAX_RATIONAL_DIGITS
        for component in components
    ):
        raise _validation_error(
            f"{label} exceeds the {MAX_RATIONAL_DIGITS}-digit bound"
        )


class IntervalExpressionNode(StrictModel):
    """One node in a bounded, non-evaluating elementary expression tree."""

    op: IntervalExpressionOp
    value: CanonicalRational | None = None
    variable: IntervalVariable | None = Field(
        default=None,
        description=(
            "Named only for box enclosure; point-enclosure variable nodes remain "
            "anonymous."
        ),
    )
    exponent: StrictInt | None = Field(
        default=None, ge=-MAX_INTEGER_EXPONENT, le=MAX_INTEGER_EXPONENT
    )
    children: tuple[IntervalExpressionNode, ...] = Field(default=(), max_length=2)

    @field_validator("children", mode="before")
    @classmethod
    def preserve_json_array_composition(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > 2:
            raise _validation_error("expression nodes may have at most two children")
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_operation_shape(self) -> Self:
        arity = {
            "const": 0,
            "var": 0,
            "neg": 1,
            "pow": 1,
            "exp": 1,
            "log": 1,
            "sqrt": 1,
            "sin": 1,
            "cos": 1,
            "add": 2,
            "sub": 2,
            "mul": 2,
            "div": 2,
        }[self.op]
        if len(self.children) != arity:
            raise _validation_error(f"{self.op} node requires exactly {arity} children")
        if self.op == "const":
            if self.value is None:
                raise _validation_error("const node requires a value")
            require_bounded_rational(
                self.value,
                max_digits=MAX_RATIONAL_DIGITS,
                label="interval-expression rational",
            )
        elif self.value is not None:
            raise _validation_error("only a const node may carry a value")
        if self.op != "var" and self.variable is not None:
            raise _validation_error("only a var node may carry a variable name")
        if self.op == "pow":
            if self.exponent is None or self.exponent == 0:
                raise _validation_error(
                    "pow node requires a nonzero bounded integer exponent"
                )
        elif self.exponent is not None:
            raise _validation_error("only a pow node may carry an exponent")
        return self


def _bounded_expression_nodes(
    expression: IntervalExpressionNode,
) -> tuple[IntervalExpressionNode, ...]:
    stack = [(expression, 1)]
    nodes: list[IntervalExpressionNode] = []
    while stack:
        node, depth = stack.pop()
        nodes.append(node)
        if depth > MAX_EXPRESSION_DEPTH:
            raise _validation_error(f"expression depth exceeds {MAX_EXPRESSION_DEPTH}")
        if len(nodes) > MAX_EXPRESSION_NODES:
            raise _validation_error(
                f"expression node count exceeds {MAX_EXPRESSION_NODES}"
            )
        stack.extend((child, depth + 1) for child in node.children)
    return tuple(nodes)


def _bound_raw_expression(expression: object) -> None:
    """Reject oversized raw trees before recursive Pydantic construction."""

    stack = [(expression, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > MAX_EXPRESSION_DEPTH:
            raise _validation_error(f"expression depth exceeds {MAX_EXPRESSION_DEPTH}")
        if count > MAX_EXPRESSION_NODES:
            raise _validation_error(
                f"expression node count exceeds {MAX_EXPRESSION_NODES}"
            )

        if isinstance(node, IntervalExpressionNode):
            children: list[object] | tuple[object, ...] = node.children
        elif isinstance(node, Mapping):
            _bound_raw_rational(node.get("value"), "interval-expression rational")
            raw_children = node.get("children")
            if not isinstance(raw_children, (list, tuple)):
                continue
            children = raw_children
        else:
            continue
        if len(children) > 2:
            raise _validation_error("expression nodes may have at most two children")
        stack.extend((child, depth + 1) for child in children)


class RationalIntervalBox(StrictModel):
    """An exact rational box with one authoritative ordered variable axis."""

    variables: tuple[IntervalVariable, ...] = Field(
        max_length=MAX_BOX_VARIABLES,
        description=(
            "Authoritative variable axis; intervals[i] is the coordinate for "
            "variables[i]."
        ),
    )
    intervals: tuple[ClosedRationalInterval, ...] = Field(
        max_length=MAX_BOX_VARIABLES,
        description="Closed rational coordinate intervals aligned to variables.",
    )

    @field_validator("variables", "intervals", mode="before")
    @classmethod
    def preserve_json_array_composition(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_BOX_VARIABLES:
            raise _validation_error(
                f"rational interval boxes admit at most {MAX_BOX_VARIABLES} coordinates"
            )
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_complete_unique_axis(self) -> Self:
        if len(self.variables) != len(self.intervals):
            raise _validation_error(
                "box variables and intervals must have the same length"
            )
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error("box variable names must be unique")
        return self


def _bound_raw_box(box: object) -> None:
    """Bound coordinate containers and endpoints before nested construction."""

    if isinstance(box, RationalIntervalBox):
        variables: object = box.variables
        intervals: object = box.intervals
    elif isinstance(box, Mapping):
        variables = box.get("variables")
        intervals = box.get("intervals")
    else:
        return
    for values in (variables, intervals):
        if isinstance(values, (list, tuple)) and len(values) > MAX_BOX_VARIABLES:
            raise _validation_error(
                f"rational interval boxes admit at most {MAX_BOX_VARIABLES} coordinates"
            )
    if not isinstance(intervals, (list, tuple)):
        return
    for interval in intervals:
        if isinstance(interval, ClosedRationalInterval):
            lower: object = interval.lower
            upper: object = interval.upper
        elif isinstance(interval, Mapping):
            lower = interval.get("lower")
            upper = interval.get("upper")
        else:
            continue
        _bound_raw_rational(lower, "expression-box endpoint")
        _bound_raw_rational(upper, "expression-box endpoint")


class IntervalExpressionDomainFailure(StrictModel):
    """The first source node whose real domain was not established."""

    node_path: tuple[StrictInt, ...] = Field(
        max_length=MAX_EXPRESSION_DEPTH - 1,
        description=(
            "Zero-based child indices from the expression root to the rejected node."
        ),
    )
    operation: Literal["div", "pow", "log", "sqrt"]
    reason: Literal[
        "DENOMINATOR_CONTAINS_ZERO",
        "NEGATIVE_POWER_BASE_CONTAINS_ZERO",
        "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
        "SQRT_ARGUMENT_NOT_NONNEGATIVE",
        "SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET",
    ]

    @field_validator("node_path", mode="before")
    @classmethod
    def preserve_json_array_composition(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) >= MAX_EXPRESSION_DEPTH:
            raise _validation_error(
                "domain-failure path exceeds the expression-depth bound"
            )
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_operation_reason_pair(self) -> Self:
        if any(index not in (0, 1) for index in self.node_path):
            raise _validation_error(
                "domain-failure paths may contain only child indices 0 or 1"
            )
        expected = {
            "div": "DENOMINATOR_CONTAINS_ZERO",
            "pow": "NEGATIVE_POWER_BASE_CONTAINS_ZERO",
            "log": "LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
            "sqrt": {
                "SQRT_ARGUMENT_NOT_NONNEGATIVE",
                "SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET",
            },
        }[self.operation]
        valid = (
            self.reason in expected
            if isinstance(expected, set)
            else self.reason == expected
        )
        if not valid:
            raise _validation_error(
                "domain-failure reason does not match its operation"
            )
        return self


@dataclass(frozen=True, slots=True)
class _RationalBounds:
    lower: Fraction
    upper: Fraction


type _BoxPreflight = _RationalBounds | IntervalExpressionDomainFailure


def _rational_box_bounds(box: RationalIntervalBox) -> dict[str, _RationalBounds]:
    return {
        variable: _RationalBounds(
            interval.lower.as_fraction(), interval.upper.as_fraction()
        )
        for variable, interval in zip(box.variables, box.intervals, strict=True)
    }


def _require_bounded_rational_components(*values: Fraction) -> None:
    """Reject produced endpoints before Fraction ordering can widen work."""

    if any(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        > MAX_BOX_INTERMEDIATE_BITS
        for value in values
    ):
        raise _validation_error(
            "expression interval intermediate exceeds the "
            f"{MAX_BOX_INTERMEDIATE_BITS}-bit rational work bound"
        )


def _bounded_rational_bounds(lower: Fraction, upper: Fraction) -> _RationalBounds:
    _require_bounded_rational_components(lower, upper)
    if lower > upper:
        raise AssertionError("internal rational interval endpoints are reversed")
    return _RationalBounds(lower, upper)


def _power_bounds(bounds: _RationalBounds, exponent: int) -> _RationalBounds:
    if exponent < 0:
        positive = _power_bounds(bounds, -exponent)
        return _bounded_rational_bounds(
            Fraction(1, 1) / positive.upper,
            Fraction(1, 1) / positive.lower,
        )

    def power(value: Fraction) -> Fraction:
        if value and any(
            component.bit_length() * exponent > MAX_BOX_INTERMEDIATE_BITS
            for component in (abs(value.numerator), value.denominator)
        ):
            raise _validation_error(
                "expression interval intermediate exceeds the "
                f"{MAX_BOX_INTERMEDIATE_BITS}-bit rational work bound"
            )
        return value**exponent

    if exponent % 2 == 1:
        return _bounded_rational_bounds(power(bounds.lower), power(bounds.upper))
    if bounds.lower >= 0:
        return _bounded_rational_bounds(power(bounds.lower), power(bounds.upper))
    if bounds.upper <= 0:
        return _bounded_rational_bounds(power(bounds.upper), power(bounds.lower))
    return _bounded_rational_bounds(
        Fraction(0), max(power(-bounds.lower), power(bounds.upper))
    )


def _bounded_power_of_four(exponent: int) -> Fraction:
    if exponent > (MAX_BOX_INTERMEDIATE_BITS - 1) // 2:
        raise _validation_error(
            "expression interval intermediate exceeds the "
            f"{MAX_BOX_INTERMEDIATE_BITS}-bit rational work bound"
        )
    return Fraction(4**exponent)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _preflight_box_unary(
    node: IntervalExpressionNode,
    bounds: _RationalBounds,
    path: tuple[int, ...],
) -> _BoxPreflight:
    if node.op == "neg":
        return _bounded_rational_bounds(-bounds.upper, -bounds.lower)
    if node.op == "pow":
        assert node.exponent is not None
        if node.exponent < 0 and bounds.lower <= 0 <= bounds.upper:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="pow",
                reason="NEGATIVE_POWER_BASE_CONTAINS_ZERO",
            )
        return _power_bounds(bounds, node.exponent)
    if node.op == "exp":
        # Since 1 < e < 4, integral powers of four give exact rational
        # magnitude bounds without evaluating a transcendental during admission.
        lower = (
            Fraction(1)
            if bounds.lower >= 0
            else Fraction(1) / _bounded_power_of_four(_ceil_fraction(-bounds.lower))
        )
        upper = (
            Fraction(1)
            if bounds.upper <= 0
            else _bounded_power_of_four(_ceil_fraction(bounds.upper))
        )
        return _bounded_rational_bounds(lower, upper)
    if node.op == "log":
        if bounds.lower <= 0:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="log",
                reason="LOG_ARGUMENT_NOT_STRICTLY_POSITIVE",
            )
        # For x > 0, 1 - 1/x <= log(x) <= x - 1.
        return _bounded_rational_bounds(
            Fraction(1) - Fraction(1) / bounds.lower,
            bounds.upper - Fraction(1),
        )
    if node.op == "sqrt":
        if bounds.lower < 0:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="sqrt",
                reason="SQRT_ARGUMENT_NOT_NONNEGATIVE",
            )
        # On [0, 1], x <= sqrt(x) <= 1; above one, 1 <= sqrt(x) <= x.
        if bounds.upper == 0:
            result = _bounded_rational_bounds(Fraction(0), Fraction(0))
        else:
            result = _bounded_rational_bounds(
                min(bounds.lower, Fraction(1)), max(bounds.upper, Fraction(1))
            )
        return result
    if node.op in ("sin", "cos"):
        return _bounded_rational_bounds(Fraction(-1), Fraction(1))
    raise AssertionError(f"unsupported unary expression operation: {node.op}")


def _preflight_box_binary(
    node: IntervalExpressionNode,
    left: _RationalBounds,
    right: _RationalBounds,
    path: tuple[int, ...],
) -> _BoxPreflight:
    if node.op == "add":
        bounds = _bounded_rational_bounds(
            left.lower + right.lower, left.upper + right.upper
        )
    elif node.op == "sub":
        bounds = _bounded_rational_bounds(
            left.lower - right.upper, left.upper - right.lower
        )
    elif node.op == "mul":
        products = (
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        )
        _require_bounded_rational_components(*products)
        bounds = _bounded_rational_bounds(min(products), max(products))
    elif node.op == "div":
        if right.lower <= 0 <= right.upper:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="div",
                reason="DENOMINATOR_CONTAINS_ZERO",
            )
        quotients = (
            left.lower / right.lower,
            left.lower / right.upper,
            left.upper / right.lower,
            left.upper / right.upper,
        )
        _require_bounded_rational_components(*quotients)
        bounds = _bounded_rational_bounds(min(quotients), max(quotients))
    else:
        raise AssertionError(f"unsupported binary expression operation: {node.op}")
    return bounds


class _IntervalExpressionBoxRequest(StrictModel):
    """Shared complete-source contract for operations over rational boxes."""

    expression: IntervalExpressionNode = Field(
        description=(
            "A non-evaluating tree of at most 64 nodes and depth 16 whose exact "
            "admission interval intermediates fit 8,192 bits."
        )
    )
    box: RationalIntervalBox
    precision_bits: StrictInt = Field(
        default=128,
        ge=32,
        le=4096,
        description="The fixed Arb work precision used for every expression node.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_tree(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if isinstance(value, Mapping):
            _bound_raw_expression(value.get("expression"))
            _bound_raw_box(value.get("box"))
        return value

    @model_validator(mode="after")
    def require_complete_named_source_axis(self) -> Self:
        nodes = _bounded_expression_nodes(self.expression)
        used_variables: set[str] = set()
        for node in nodes:
            if node.op != "var":
                continue
            if node.variable is None:
                raise _validation_error("box-enclosure variable nodes must be named")
            used_variables.add(node.variable)
        box_variables = set(self.box.variables)
        missing = used_variables - box_variables
        if missing:
            raise _validation_error(
                "expression variables are missing from the box: "
                + ", ".join(sorted(missing))
            )
        unused = box_variables - used_variables
        if unused:
            raise _validation_error(
                "box variables are unused by the expression: "
                + ", ".join(sorted(unused))
            )
        return self


class ExactDyadic(StrictModel):
    """The exact value ``mantissa * 2**exponent``."""

    mantissa: str = Field(
        pattern=r"^-?(?:0|[1-9][0-9]*)$", max_length=MAX_DYADIC_MANTISSA_DIGITS
    )
    exponent: StrictInt = Field(ge=-MAX_DYADIC_EXPONENT, le=MAX_DYADIC_EXPONENT)

    @model_validator(mode="after")
    def require_canonical_binary_form(self) -> Self:
        mantissa = int(self.mantissa)
        if mantissa == 0 and self.exponent != 0:
            raise _validation_error("canonical dyadic zero must have exponent 0")
        if mantissa != 0 and mantissa % 2 == 0:
            raise _validation_error("canonical nonzero dyadic mantissa must be odd")
        return self

    def as_fraction(self) -> Fraction:
        mantissa = Fraction(int(self.mantissa))
        if self.exponent >= 0:
            return mantissa * Fraction(2**self.exponent, 1)
        return mantissa / Fraction(2 ** (-self.exponent), 1)

    def compare(self, other: ExactDyadic) -> int:
        """Compare two dyadics without materializing either power of two."""

        left = int(self.mantissa)
        right = int(other.mantissa)
        if left == 0 or right == 0 or (left < 0) != (right < 0):
            return (left > right) - (left < right)

        left_magnitude = abs(left)
        right_magnitude = abs(right)
        left_top_bit = left_magnitude.bit_length() + self.exponent
        right_top_bit = right_magnitude.bit_length() + other.exponent
        if left_top_bit != right_top_bit:
            magnitude_order = (left_top_bit > right_top_bit) - (
                left_top_bit < right_top_bit
            )
        elif self.exponent >= other.exponent:
            magnitude_order = (
                (left_magnitude << (self.exponent - other.exponent)) > right_magnitude
            ) - ((left_magnitude << (self.exponent - other.exponent)) < right_magnitude)
        else:
            magnitude_order = (
                left_magnitude > (right_magnitude << (other.exponent - self.exponent))
            ) - (left_magnitude < (right_magnitude << (other.exponent - self.exponent)))
        return magnitude_order if left > 0 else -magnitude_order


class DyadicClosedInterval(StrictModel):
    """One closed interval with exact dyadic endpoints."""

    lower: ExactDyadic
    upper: ExactDyadic

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        if self.lower.compare(self.upper) > 0:
            raise _validation_error(
                "dyadic interval lower endpoint exceeds upper endpoint"
            )
        return self
