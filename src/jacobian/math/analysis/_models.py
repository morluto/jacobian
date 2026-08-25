"""Typed contracts for rigorous real-function enclosures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.geometry.boxes.values import RationalClosedInterval


def _validation_error(message: str) -> PydanticCustomError:
    """Project analysis-model invariants through a stable owner code."""

    return PydanticCustomError("analysis.invariant", message)


MAX_RATIONAL_DIGITS = 128
MAX_EXPRESSION_DEPTH = 16
MAX_EXPRESSION_NODES = 64
MAX_INTEGER_EXPONENT = 64
MAX_BOX_VARIABLES = 8
MAX_SECOND_JET_VARIABLES = MAX_BOX_VARIABLES
MAX_SECOND_JET_RESULT_INTERVALS = (
    1
    + MAX_SECOND_JET_VARIABLES
    + MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2
)


def _second_jet_node_arithmetic_units(dimension: int) -> int:
    """Bound the scalar Arb steps of one source node's dense second-order jet.

    Every jet carries a value, gradient, and dense Hessian, so one node's
    arithmetic grows quadratically in the jet dimension.  A division is the
    most expensive node: one reciprocal power jet followed by one product jet
    costs about ``10 * dimension**2 + 4 * dimension`` scalar steps, and the
    padded constant covers transcendental chain-rule values.  At three
    variables this reproduces the former flat 128-unit bound.
    """
    return 10 * dimension * dimension + 4 * dimension + 16


# Twice the former 64-node x 128-unit envelope.  Every previously accepted
# three-variable request stays admitted while the dimension-derived charge
# admits small full-axis jets such as an eight-variable affine form.
MAX_SECOND_JET_WORK_UNITS = 16_384
# Cap every retained exact preflight numerator and denominator at twice the
# maximum Arb work precision.  One Fraction binary operation or comparison can
# transiently combine two such components, so its temporary size is separately
# bounded by the derived double-width ceiling below.
MAX_BOX_INTERMEDIATE_BITS = 8_192
MAX_BOX_PREFLIGHT_TEMPORARY_BITS = 2 * MAX_BOX_INTERMEDIATE_BITS + 1
MAX_DYADIC_EXPONENT = 2**53 - 1
MAX_DYADIC_MANTISSA_DIGITS = 1_235
MAX_POINT_CHECK_DYADIC_EXPONENT = 8_192
MAX_POINT_CHECK_LOG_TERMS = 128
MAX_POINT_CHECK_FRACTION_BITS = 131_072
MAX_POINT_CHECK_FRACTION_UPDATES = 4 * MAX_POINT_CHECK_LOG_TERMS
MAX_POINT_CHECK_OUTPUT_BYTES = 4_096

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
type IntervalExpressionBoxEnclosureStatus = Literal[
    "ENCLOSED", "DOMAIN_UNPROVEN", "BACKEND_ERROR"
]
type IntervalExpressionSecondJetEnclosureStatus = Literal[
    "ENCLOSED", "DOMAIN_UNPROVEN", "BACKEND_ERROR"
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


class IntervalExpressionEnclosureRequest(StrictModel):
    """Evaluate a bounded expression at one exact rational argument using Arb."""

    expression: IntervalExpressionNode
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="before")
    @classmethod
    def bound_raw_tree(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if isinstance(value, Mapping):
            _bound_raw_expression(value.get("expression"))
            _bound_raw_rational(value.get("argument"), "interval-enclosure argument")
        return value

    @model_validator(mode="after")
    def require_bounded_tree(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="interval-enclosure argument",
        )
        nodes = _bounded_expression_nodes(self.expression)
        if any(node.op == "var" and node.variable is not None for node in nodes):
            raise _validation_error(
                "point-enclosure variable nodes must remain anonymous"
            )
        return self


class RationalIntervalBox(StrictModel):
    """An exact rational box with one authoritative ordered variable axis."""

    variables: tuple[IntervalVariable, ...] = Field(
        max_length=MAX_BOX_VARIABLES,
        description=(
            "Authoritative variable axis; intervals[i] is the coordinate for "
            "variables[i]."
        ),
    )
    intervals: tuple[RationalClosedInterval, ...] = Field(
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
        if isinstance(interval, RationalClosedInterval):
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


def _preflight_box_expression(
    node: IntervalExpressionNode,
    variables: dict[str, _RationalBounds],
    path: tuple[int, ...] = (),
) -> _BoxPreflight:
    """Bound exact admission work and locate the first domain obstruction."""

    if node.op == "const":
        assert node.value is not None
        value = node.value.as_fraction()
        return _bounded_rational_bounds(value, value)
    if node.op == "var":
        assert node.variable is not None
        return variables[node.variable]

    children: list[_BoxPreflight] = []
    for index, child_node in enumerate(node.children):
        child = _preflight_box_expression(child_node, variables, (*path, index))
        if isinstance(child, IntervalExpressionDomainFailure):
            return child
        children.append(child)

    left = children[0]
    assert isinstance(left, _RationalBounds)
    if len(children) == 1:
        return _preflight_box_unary(node, left, path)

    right = children[1]
    assert isinstance(right, _RationalBounds)
    return _preflight_box_binary(node, left, right, path)


class IntervalExpressionBoxEnclosureRequest(StrictModel):
    """Enclose one named-variable expression on a complete rational box."""

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
    def require_complete_bounded_source(self) -> Self:
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
        _preflight_box_expression(self.expression, _rational_box_bounds(self.box))
        return self


def _preflight_second_jet_expression(
    node: IntervalExpressionNode,
    variables: dict[str, _RationalBounds],
    path: tuple[int, ...] = (),
) -> _BoxPreflight:
    """Prove the source tree is twice differentiable on the complete box."""

    if node.op == "const":
        assert node.value is not None
        value = node.value.as_fraction()
        return _bounded_rational_bounds(value, value)
    if node.op == "var":
        assert node.variable is not None
        return variables[node.variable]

    children: list[_BoxPreflight] = []
    for index, child_node in enumerate(node.children):
        child = _preflight_second_jet_expression(child_node, variables, (*path, index))
        if isinstance(child, IntervalExpressionDomainFailure):
            return child
        children.append(child)

    left = children[0]
    assert isinstance(left, _RationalBounds)
    if len(children) == 1:
        if node.op == "sqrt" and left.lower <= 0:
            return IntervalExpressionDomainFailure(
                node_path=path,
                operation="sqrt",
                reason="SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET",
            )
        return _preflight_box_unary(node, left, path)

    right = children[1]
    assert isinstance(right, _RationalBounds)
    return _preflight_box_binary(node, left, right, path)


class IntervalExpressionSecondJetEnclosureRequest(
    IntervalExpressionBoxEnclosureRequest
):
    """Enclose an elementary expression and all derivatives through order two."""

    @model_validator(mode="after")
    def require_small_twice_differentiable_source(self) -> Self:
        dimension = len(self.box.variables)
        result_intervals = 1 + dimension + dimension * (dimension + 1) // 2
        if result_intervals > MAX_SECOND_JET_RESULT_INTERVALS:
            raise AssertionError(
                "second-jet result interval accounting is inconsistent"
            )
        work_units = len(
            _bounded_expression_nodes(self.expression)
        ) * _second_jet_node_arithmetic_units(dimension)
        if work_units > MAX_SECOND_JET_WORK_UNITS:
            raise _validation_error(
                f"second-jet forward arithmetic work of {work_units} scalar "
                f"units exceeds its {MAX_SECOND_JET_WORK_UNITS}-unit bound at "
                "this dimension"
            )
        _preflight_second_jet_expression(
            self.expression, _rational_box_bounds(self.box)
        )
        return self


class IntervalExpressionEnclosureResult(StrictModel):
    status: Literal[
        "ENCLOSED",
        "DOMAIN_ERROR",
        "PRECISION_INSUFFICIENT",
        "NONFINITE",
        "OUTPUT_MAGNITUDE_EXCEEDED",
    ]
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise _validation_error(
                "only an enclosed result may carry dyadic endpoints"
            )
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise _validation_error(
                "a non-enclosure cannot claim accuracy or exactness"
            )
        if enclosed:
            assert self.lower is not None and self.upper is not None
            if self.lower.compare(self.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.exact != (self.relative_accuracy_bits is None):
                raise _validation_error(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self


class RealUnaryFunction(StrEnum):
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    SIN = "SIN"
    COS = "COS"


class ArbPointEnclosureRequest(StrictModel):
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="validated-analysis rational",
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


type PointEnclosureCheckOutcome = Literal["ACCEPTED", "REJECTED", "NON_RESULT"]


class ClaimedPointEnclosure(StrictModel):
    """One claimed enclosure of a real function value by exact dyadic endpoints.

    This is the domain-owned canonical value shared by the Arb producer and
    the independent checker, so a serialized claim crosses the consumer
    boundary unchanged. Endpoint order is deliberately not validated here: a
    reversed claim is a checkable mathematical statement, not an invalid one.
    """

    function: RealUnaryFunction = Field(
        description="Real function whose value the endpoints claim to enclose."
    )
    argument: CanonicalRational = Field(
        description="Exact reduced rational argument with at most 128 digits per component."
    )
    precision_bits: StrictInt = Field(
        ge=32,
        le=4096,
        description=(
            "Precision metadata retained from the source computation; it does "
            "not promise that an independent replay resolves the claim at "
            "that precision."
        ),
    )
    lower: ExactDyadic = Field(description="Claimed inclusive lower endpoint.")
    upper: ExactDyadic = Field(description="Claimed inclusive upper endpoint.")

    @model_validator(mode="after")
    def bound_claim_source(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="claimed point-enclosure rational",
        )
        return self


def _preflight_point_check_source(data: object) -> object:
    """Reject oversized raw source scalars before canonical integer parsing."""

    if not isinstance(data, dict):
        return data
    enclosure = data.get("enclosure")
    if isinstance(enclosure, ClaimedPointEnclosure):
        raw_components: tuple[object, ...] = (
            enclosure.argument.num,
            enclosure.argument.den,
        )
    elif isinstance(enclosure, dict):
        argument = enclosure.get("argument")
        if isinstance(argument, CanonicalRational):
            raw_components = (argument.num, argument.den)
        elif isinstance(argument, dict):
            raw_components = (argument.get("num"), argument.get("den"))
        else:
            raw_components = ()
    else:
        raw_components = ()
    if any(
        isinstance(component, str) and len(component.lstrip("-")) > MAX_RATIONAL_DIGITS
        for component in raw_components
    ):
        raise _validation_error(
            "point-enclosure checker raw rational component exceeds the "
            f"{MAX_RATIONAL_DIGITS}-digit bound"
        )
    return data


def _point_check_fraction_bound_bits(argument: CanonicalRational) -> int:
    """Bound LOG/SQRT intermediates, including claim comparisons and replay."""

    numerator, denominator = argument.as_integer_ratio()
    source_bits = max(abs(numerator).bit_length(), denominator.bit_length())
    transformed_bits = source_bits + 2
    odd_denominator_bits = (2 * MAX_POINT_CHECK_LOG_TERMS + 1).bit_length()

    # A common denominator for one upper atanh bound divides
    # b**(2*n-1) * lcm(1, 3, ..., 2*n-1) * (2*n+1) * (b*b-a*a).
    # Combining log(y) with k*log(2) adds the independent powers of 3 but
    # shares the odd-denominator factors.  The coefficient k has at most
    # source_bits.bit_length() bits after exact power-of-two range reduction.
    combined_log_bits = (
        (2 * MAX_POINT_CHECK_LOG_TERMS - 1) * (transformed_bits + 2)
        + 2 * transformed_bits
        + (MAX_POINT_CHECK_LOG_TERMS + 1) * odd_denominator_bits
        + source_bits.bit_length()
        + 8
    )
    endpoint_bits = 4 * MAX_DYADIC_MANTISSA_DIGITS + MAX_POINT_CHECK_DYADIC_EXPONENT
    log_comparison_bits = combined_log_bits + endpoint_bits + 32
    sqrt_comparison_bits = 2 * endpoint_bits + source_bits + 8
    return max(log_comparison_bits, sqrt_comparison_bits)


class PointEnclosureCheckRequest(StrictModel):
    """Check one claimed LOG or SQRT enclosure by exact independent replay.

    The claimed enclosure is one canonical ``ClaimedPointEnclosure`` accepted
    unchanged from its source. Rational components have at most 128 decimal
    digits. Claimed dyadic exponents must lie in -8192..8192; reversed or
    mathematically invalid intervals remain valid claims and produce typed
    checker outcomes. Only LOG and SQRT claims are admitted. LOG replay uses
    at most 128 terms per series, about 400 worst-case bits after range
    reduction, so tighter claims can produce NON_RESULT even when their
    retained precision metadata is larger.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_log_term_bound": MAX_POINT_CHECK_LOG_TERMS,
            "point_check_fraction_intermediate_bit_bound": (
                MAX_POINT_CHECK_FRACTION_BITS
            ),
            "point_check_producer_replay_term_update_bound": (
                MAX_POINT_CHECK_FRACTION_UPDATES
            ),
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    enclosure: ClaimedPointEnclosure = Field(
        description=(
            "Canonical claimed enclosure retained verbatim from its source; "
            "only LOG and SQRT functions are admitted."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        return _preflight_point_check_source(data)

    @model_validator(mode="after")
    def preflight_exact_checker(self) -> Self:
        if self.enclosure.function not in (
            RealUnaryFunction.LOG,
            RealUnaryFunction.SQRT,
        ):
            raise _validation_error(
                "point-enclosure checker replays only LOG and SQRT claims"
            )
        if any(
            abs(endpoint.exponent) > MAX_POINT_CHECK_DYADIC_EXPONENT
            for endpoint in (self.enclosure.lower, self.enclosure.upper)
        ):
            raise _validation_error(
                "point-enclosure checker dyadic exponent exceeds the "
                f"+/-{MAX_POINT_CHECK_DYADIC_EXPONENT} bound"
            )
        if (
            _point_check_fraction_bound_bits(self.enclosure.argument)
            > MAX_POINT_CHECK_FRACTION_BITS
        ):
            raise _validation_error(
                "point-enclosure checker exact rational work exceeds the "
                f"{MAX_POINT_CHECK_FRACTION_BITS}-bit intermediate bound"
            )
        return self


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


class FirstPartialEnclosure(StrictModel):
    """One first-partial enclosure bound to the authoritative source axis."""

    variable: IntervalVariable
    enclosure: DyadicClosedInterval


class HessianEntryEnclosure(StrictModel):
    """One canonical upper-triangular Hessian entry over the source box."""

    first_variable: IntervalVariable
    second_variable: IntervalVariable
    enclosure: DyadicClosedInterval


class IntervalExpressionSecondJetEnclosureResult(
    IntervalExpressionSecondJetEnclosureRequest
):
    """A source-bound rigorous value, gradient, and symmetric Hessian enclosure.

    For ``ENCLOSED``, full validation recomputes the canonical jet claim from
    its expression, axis, and source box.  ``DOMAIN_UNPROVEN`` replays its
    deterministic first-obstruction evidence.  ``BACKEND_ERROR`` asserts no
    enclosure conclusion at all, so it is validated structurally: rerunning
    Arb would reject the operation's own serialized result whenever a
    transient backend condition does not recur.
    """

    status: IntervalExpressionSecondJetEnclosureStatus
    value: DyadicClosedInterval | None = None
    gradient: tuple[FirstPartialEnclosure, ...] = Field(
        default=(), max_length=MAX_SECOND_JET_VARIABLES
    )
    hessian: tuple[HessianEntryEnclosure, ...] = Field(
        default=(),
        max_length=MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2,
    )
    domain_failure: IntervalExpressionDomainFailure | None = None
    method: Literal["ARB_FORWARD_SECOND_ORDER_JET"] = "ARB_FORWARD_SECOND_ORDER_JET"
    detail: str = Field(min_length=1, max_length=1024)

    @field_validator("gradient", mode="before")
    @classmethod
    def preserve_gradient_json_array_composition(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_SECOND_JET_VARIABLES:
            raise _validation_error(
                "second-jet result exceeds its declared interval bound"
            )
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("hessian", mode="before")
    @classmethod
    def preserve_hessian_json_array_composition(cls, value: object) -> object:
        maximum = MAX_SECOND_JET_VARIABLES * (MAX_SECOND_JET_VARIABLES + 1) // 2
        if isinstance(value, (list, tuple)) and len(value) > maximum:
            raise _validation_error(
                "second-jet result exceeds its declared interval bound"
            )
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def bind_second_jet_to_source(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        expected_pairs = tuple(
            (first, second)
            for first_index, first in enumerate(self.box.variables)
            for second in self.box.variables[first_index:]
        )
        if enclosed != (self.value is not None):
            raise _validation_error(
                "only an enclosed second jet may carry a value enclosure"
            )
        if enclosed:
            if tuple(entry.variable for entry in self.gradient) != self.box.variables:
                raise _validation_error(
                    "gradient entries must match the ordered source axis"
                )
            if (
                tuple(
                    (entry.first_variable, entry.second_variable)
                    for entry in self.hessian
                )
                != expected_pairs
            ):
                raise _validation_error(
                    "Hessian entries must be the canonical upper triangle of the source axis"
                )
            if self.domain_failure is not None:
                raise _validation_error(
                    "an enclosed second jet cannot carry a domain failure"
                )
        else:
            if self.value is not None or self.gradient or self.hessian:
                raise _validation_error(
                    "a non-enclosed second jet cannot carry partial enclosures"
                )
            domain_unproven = self.status == "DOMAIN_UNPROVEN"
            if domain_unproven != (self.domain_failure is not None):
                raise _validation_error(
                    "domain-failure evidence must agree with DOMAIN_UNPROVEN status"
                )
            if self.status == "BACKEND_ERROR":
                return self

        from jacobian.math.analysis._operations import _second_jet_enclosure

        request = IntervalExpressionSecondJetEnclosureRequest.model_construct(
            expression=self.expression,
            box=self.box,
            precision_bits=self.precision_bits,
        )
        if self != _second_jet_enclosure(request):
            raise _validation_error(
                "second-jet enclosure does not replay from its expression, axis, and source box"
            )
        return self


class PointEnclosureCheckResult(StrictModel):
    """A source-bound checker outcome replayed during result validation.

    ACCEPTED means the independently proved interval is contained in the
    claim. REJECTED covers an invalid real-domain or reversed claim, or a claim
    proved disjoint from the true value. NON_RESULT means the independent LOG
    enclosure still partially overlaps the claim after 128 series terms.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "point_check_output_byte_bound": MAX_POINT_CHECK_OUTPUT_BYTES,
        }
    )

    enclosure: ClaimedPointEnclosure
    outcome: PointEnclosureCheckOutcome = Field(
        description=(
            "ACCEPTED when the independent enclosure is contained in the "
            "claim; REJECTED for invalid or provably excluding claims; "
            "NON_RESULT for unresolved partial overlap at the LOG term cap."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source(cls, data: object) -> object:
        data = canonicalize_json_containers(data)
        return _preflight_point_check_source(data)

    @model_validator(mode="after")
    def replay_outcome(self) -> Self:
        from jacobian.math.analysis._point_enclosure_check import (
            point_enclosure_check_outcome,
        )

        request = PointEnclosureCheckRequest(enclosure=self.enclosure)
        if self.outcome != point_enclosure_check_outcome(request):
            raise _validation_error(
                "outcome must equal the deterministic enclosure check for the retained source"
            )
        return self


class IntervalExpressionBoxEnclosureResult(IntervalExpressionBoxEnclosureRequest):
    """A replayable enclosure bound to its expression, axis, and source box.

    For ``ENCLOSED``, every defined real source-box value lies between the two
    exact dyadic endpoints.  Full validation recomputes that canonical claim.
    ``DOMAIN_UNPROVEN`` replays its deterministic first-obstruction evidence.
    ``BACKEND_ERROR`` asserts no enclosure conclusion at all, so it is
    validated structurally: rerunning Arb would reject the operation's own
    serialized result whenever a transient backend condition does not recur.
    """

    status: IntervalExpressionBoxEnclosureStatus
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    domain_failure: IntervalExpressionDomainFailure | None = None
    method: Literal["ARB_NATURAL_INTERVAL_EXTENSION"] = "ARB_NATURAL_INTERVAL_EXTENSION"
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_source(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise _validation_error(
                "only an enclosed result may carry dyadic endpoints"
            )
        if enclosed:
            assert self.lower is not None and self.upper is not None
            if self.lower.compare(self.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.domain_failure is not None:
                raise _validation_error(
                    "an enclosed result cannot carry a domain failure"
                )
        else:
            domain_unproven = self.status == "DOMAIN_UNPROVEN"
            if domain_unproven != (self.domain_failure is not None):
                raise _validation_error(
                    "domain-failure evidence must agree with DOMAIN_UNPROVEN status"
                )
            if self.status == "BACKEND_ERROR":
                return self

        from jacobian.math.analysis._operations import _box_expression_enclosure

        request = IntervalExpressionBoxEnclosureRequest.model_construct(
            expression=self.expression,
            box=self.box,
            precision_bits=self.precision_bits,
        )
        if self != _box_expression_enclosure(request):
            raise _validation_error(
                "box enclosure does not replay from its expression, axis, and source box"
            )
        return self


class ArbPointEnclosureResult(ArbPointEnclosureRequest):
    """A source-bound Arb ball enclosure of one real function value.

    Every outcome retains the request's function, argument, and precision;
    ``enclosure`` carries the canonical ``ClaimedPointEnclosure`` only when
    ``status`` is ``ENCLOSED``, and must restate that retained source.
    """

    status: Literal[
        "ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR", "OUTPUT_MAGNITUDE_EXCEEDED"
    ]
    enclosure: ClaimedPointEnclosure | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.enclosure is not None):
            raise _validation_error(
                "only an enclosed result may carry the point enclosure"
            )
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise _validation_error(
                "a non-enclosure cannot claim accuracy or exactness"
            )
        if enclosed:
            enclosure = self.enclosure
            assert enclosure is not None
            if (
                enclosure.function,
                enclosure.argument,
                enclosure.precision_bits,
            ) != (self.function, self.argument, self.precision_bits):
                raise _validation_error(
                    "the enclosure must restate the retained request"
                )
            if enclosure.lower.compare(enclosure.upper) > 0:
                raise _validation_error(
                    "enclosure lower endpoint exceeds upper endpoint"
                )
            if self.exact != (self.relative_accuracy_bits is None):
                raise _validation_error(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self
