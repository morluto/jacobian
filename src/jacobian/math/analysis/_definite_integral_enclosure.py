"""Rigorous definite-integral enclosures over exact rational partitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from time import monotonic
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    format_canonical_integer,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._box_enclosure import (
    _BoxEvaluationFailure,
    _evaluate_box_expression,
    _preflight_box_expression,
)
from jacobian.math.analysis._models import (
    MAX_BOX_INTERMEDIATE_BITS,
    MAX_DYADIC_MANTISSA_DIGITS,
    MAX_EXPRESSION_NODES,
    MAX_RATIONAL_DIGITS,
    DyadicClosedInterval,
    ExactDyadic,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    RationalIntervalBox,
    _bounded_expression_nodes,
    _BoxPreflight,
    _IntervalExpressionBoxRequest,
    _rational_box_bounds,
    _RationalBounds,
    _validation_error,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval

MAX_DEFINITE_INTEGRAL_LEAVES = 1_024
MAX_DEFINITE_INTEGRAL_DEPTH = MAX_DEFINITE_INTEGRAL_LEAVES - 1
MAX_DEFINITE_INTEGRAL_SUBPROBLEMS = 2 * MAX_DEFINITE_INTEGRAL_LEAVES - 1
MAX_DEFINITE_INTEGRAL_NODE_WORK = (
    2 * MAX_EXPRESSION_NODES * MAX_DEFINITE_INTEGRAL_SUBPROBLEMS
)
# This conservative node-bit ceiling admits a full 64-node/1,024-leaf request
# through 2,048 bits, or a 32-node request through the 4,096-bit backend limit.
MAX_DEFINITE_INTEGRAL_PRECISION_WORK = 268_435_456
MAX_DEFINITE_INTEGRAL_SELECTION_COMPARISONS = (
    MAX_DEFINITE_INTEGRAL_LEAVES * (MAX_DEFINITE_INTEGRAL_LEAVES - 1) // 2
)
MAX_DEFINITE_INTEGRAL_SUMMATION_UNITS = MAX_DEFINITE_INTEGRAL_LEAVES * (
    MAX_DEFINITE_INTEGRAL_LEAVES + 1
)
MAX_DEFINITE_INTEGRAL_WALL_SECONDS = 120
MAX_DEFINITE_INTEGRAL_RESULT_BYTES = CanonicalLimits().max_output_bytes

# A retained rational preflight bound has at most 8,192 component bits. A source
# endpoint has at most 128 decimal digits (< 4 * 128 bits), a midpoint path adds
# at most 1,023 denominator bits, and a normalized 4,096-bit endpoint may shift
# by its precision. The final sum adds at most ceil(log2(1,024)) magnitude bits.
# This owner-local ceiling is therefore strictly below the ambient ExactDyadic
# wire exponent and bounds every Fraction shift performed by result validation.
MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT = (
    MAX_BOX_INTERMEDIATE_BITS
    + 4 * MAX_RATIONAL_DIGITS
    + MAX_DEFINITE_INTEGRAL_DEPTH
    + 4_096
    + (MAX_DEFINITE_INTEGRAL_LEAVES - 1).bit_length()
    + 4
)
MAX_DEFINITE_INTEGRAL_ACCUMULATOR_BITS = (
    4_096
    + 2 * MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT
    + (MAX_DEFINITE_INTEGRAL_LEAVES - 1).bit_length()
)


class DefiniteIntegralEnclosureRequest(_IntervalExpressionBoxRequest):
    """Enclose one definite integral over an exact one-dimensional box."""

    target_width: ExactDyadic = Field(
        description=(
            "A nonnegative exact dyadic target for integral upper-lower; zero "
            "requests an exact singleton enclosure."
        )
    )
    max_leaves: StrictInt = Field(
        default=32,
        ge=1,
        le=MAX_DEFINITE_INTEGRAL_LEAVES,
        description=(
            "Maximum leaves in the exact binary midpoint partition. This is the "
            "only caller-selected partition budget."
        ),
    )
    wall_seconds: StrictInt = Field(
        default=30,
        ge=1,
        le=MAX_DEFINITE_INTEGRAL_WALL_SECONDS,
        description=(
            "Shared owner deadline in seconds, measured from dispatch start (or "
            "native-function entry) through result construction."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def preserve_raw_request_limits(cls, value: object) -> object:
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_complete_named_source_axis(self) -> Self:
        if len(self.box.variables) != 1:
            raise _validation_error(
                "definite integration requires exactly one named source variable"
            )
        used_variables: set[str] = set()
        for node in _bounded_expression_nodes(self.expression):
            if node.op != "var":
                continue
            if node.variable is None:
                raise _validation_error(
                    "definite-integral variable nodes must be named"
                )
            used_variables.add(node.variable)
        if used_variables and used_variables != set(self.box.variables):
            raise _validation_error(
                "the expression's named variable must match the integration axis"
            )
        if int(self.target_width.mantissa) < 0:
            raise _validation_error(
                "definite-integral target width must be nonnegative"
            )
        return self


class _IntegralLeafPath(StrictModel):
    """One canonical address in the exact midpoint partition."""

    path: tuple[StrictInt, ...] = Field(
        max_length=MAX_DEFINITE_INTEGRAL_DEPTH,
        description=(
            "Binary child choices from the source interval; 0 is the lower and 1 "
            "the upper exact-midpoint child."
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def preserve_bounded_path(cls, value: object) -> object:
        if (
            isinstance(value, (list, tuple))
            and len(value) > MAX_DEFINITE_INTEGRAL_DEPTH
        ):
            raise _validation_error(
                "definite-integral leaf path exceeds the partition-depth bound"
            )
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_binary_path(self) -> Self:
        if any(bit not in (0, 1) for bit in self.path):
            raise _validation_error("definite-integral leaf paths contain only 0 and 1")
        return self


class DefiniteIntegralEnclosedLeaf(_IntegralLeafPath):
    """One positive-measure leaf with a sound range and integral contribution."""

    status: Literal["ENCLOSED"] = "ENCLOSED"
    range_enclosure: DyadicClosedInterval
    contribution: DyadicClosedInterval


class DefiniteIntegralDomainUnprovenLeaf(_IntegralLeafPath):
    """One leaf whose exact natural extension did not prove the real domain."""

    status: Literal["DOMAIN_UNPROVEN"] = "DOMAIN_UNPROVEN"
    domain_failure: IntervalExpressionDomainFailure


class DefiniteIntegralZeroMeasureLeaf(_IntegralLeafPath):
    """The unique degenerate source interval, whose integral is exactly zero."""

    status: Literal["ZERO_MEASURE"] = "ZERO_MEASURE"
    contribution: DyadicClosedInterval


type DefiniteIntegralLeaf = Annotated[
    DefiniteIntegralEnclosedLeaf
    | DefiniteIntegralDomainUnprovenLeaf
    | DefiniteIntegralZeroMeasureLeaf,
    Field(discriminator="status"),
]

type DefiniteIntegralConcludedLeaf = Annotated[
    DefiniteIntegralEnclosedLeaf | DefiniteIntegralZeroMeasureLeaf,
    Field(discriminator="status"),
]


def _preserve_bounded_leaf_payload(value: object) -> object:
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_DEFINITE_INTEGRAL_LEAVES:
            raise _validation_error("definite-integral result exceeds its leaf bound")
        for raw_leaf in value:
            if isinstance(raw_leaf, Mapping):
                path = raw_leaf.get("path")
                if (
                    isinstance(path, (list, tuple))
                    and len(path) > MAX_DEFINITE_INTEGRAL_DEPTH
                ):
                    raise _validation_error(
                        "definite-integral leaf path exceeds the partition-depth bound"
                    )
    return tuple(value) if isinstance(value, list) else value


class DefiniteIntegralTargetMet(StrictModel):
    """A sound final integral enclosure no wider than the requested target."""

    status: Literal["TARGET_MET"] = "TARGET_MET"
    enclosure: DyadicClosedInterval
    leaves: tuple[DefiniteIntegralConcludedLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_DEFINITE_INTEGRAL_LEAVES,
        description=(
            "Canonical path-ordered complete midpoint cover. Every retained leaf "
            "carries its exact outward dyadic integral contribution."
        ),
    )

    @field_validator("leaves", mode="before")
    @classmethod
    def preserve_bounded_leaves(cls, value: object) -> object:
        return _preserve_bounded_leaf_payload(value)


class DefiniteIntegralBudgetExhausted(StrictModel):
    """A sound complete cover whose leaf budget did not meet the target."""

    status: Literal["BUDGET_EXHAUSTED"] = "BUDGET_EXHAUSTED"
    reason: Literal["MAX_LEAVES"] = "MAX_LEAVES"
    enclosure: DyadicClosedInterval
    leaves: tuple[DefiniteIntegralConcludedLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_DEFINITE_INTEGRAL_LEAVES,
        description=(
            "Canonical path-ordered complete midpoint cover. Every retained leaf "
            "carries its exact outward dyadic integral contribution."
        ),
    )

    @field_validator("leaves", mode="before")
    @classmethod
    def preserve_bounded_leaves(cls, value: object) -> object:
        return _preserve_bounded_leaf_payload(value)


class DefiniteIntegralDomainUnproven(StrictModel):
    """A complete cover with no global conclusion because a leaf is unproven."""

    status: Literal["DOMAIN_UNPROVEN"] = "DOMAIN_UNPROVEN"
    leaves: tuple[DefiniteIntegralLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_DEFINITE_INTEGRAL_LEAVES,
        description=(
            "Canonical path-ordered complete midpoint cover retaining every sound "
            "contribution and every local domain-failure diagnostic."
        ),
    )

    @field_validator("leaves", mode="before")
    @classmethod
    def preserve_bounded_leaves(cls, value: object) -> object:
        return _preserve_bounded_leaf_payload(value)


type DefiniteIntegralOutcome = Annotated[
    DefiniteIntegralTargetMet
    | DefiniteIntegralBudgetExhausted
    | DefiniteIntegralDomainUnproven,
    Field(discriminator="status"),
]


def _normalize_dyadic_pair(mantissa: int, exponent: int) -> tuple[int, int]:
    if mantissa == 0:
        return 0, 0
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return mantissa, exponent


def _exact_dyadic(mantissa: int, exponent: int) -> ExactDyadic:
    mantissa, exponent = _normalize_dyadic_pair(mantissa, exponent)
    if abs(exponent) > MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT:
        raise _validation_error(
            "definite-integral dyadic exponent exceeds the admitted source bound"
        )
    return ExactDyadic(mantissa=format_canonical_integer(mantissa), exponent=exponent)


def _dyadic_fraction(value: ExactDyadic) -> Fraction:
    if abs(value.exponent) > MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT:
        raise _validation_error(
            "definite-integral dyadic exponent exceeds the admitted source bound"
        )
    return value.as_fraction()


def _floor_log2_abs(value: Fraction) -> int:
    """Return floor(log2(abs(value))) without floating-point conversion."""

    if value == 0:
        raise ValueError("zero has no binary magnitude")
    numerator = abs(value.numerator)
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << (-exponent) < denominator:
        exponent -= 1
    return exponent


def _round_fraction_outward(
    value: Fraction, precision_bits: int, *, toward_positive: bool
) -> ExactDyadic:
    """Round one rational outward to a deterministic significant-bit dyadic."""

    if value == 0:
        return ExactDyadic(mantissa="0", exponent=0)
    exponent = _floor_log2_abs(value) - (precision_bits - 1)
    if exponent >= 0:
        scaled_numerator = value.numerator
        scaled_denominator = value.denominator << exponent
    else:
        scaled_numerator = value.numerator << (-exponent)
        scaled_denominator = value.denominator
    if toward_positive:
        mantissa = -((-scaled_numerator) // scaled_denominator)
    else:
        mantissa = scaled_numerator // scaled_denominator
    return _exact_dyadic(mantissa, exponent)


def _interval_width(interval: ClosedRationalInterval) -> Fraction:
    return interval.upper.as_fraction() - interval.lower.as_fraction()


def _enclosure_width(enclosure: DyadicClosedInterval) -> Fraction:
    return _dyadic_fraction(enclosure.upper) - _dyadic_fraction(enclosure.lower)


def _leaf_contribution(
    interval: ClosedRationalInterval,
    range_enclosure: DyadicClosedInterval,
    precision_bits: int,
) -> DyadicClosedInterval:
    width = _interval_width(interval)
    lower = width * _dyadic_fraction(range_enclosure.lower)
    upper = width * _dyadic_fraction(range_enclosure.upper)
    return DyadicClosedInterval(
        lower=_round_fraction_outward(lower, precision_bits, toward_positive=False),
        upper=_round_fraction_outward(upper, precision_bits, toward_positive=True),
    )


def _summed_enclosure(
    contributions: tuple[DyadicClosedInterval, ...], precision_bits: int
) -> DyadicClosedInterval:
    lower = sum((_dyadic_fraction(value.lower) for value in contributions), Fraction())
    upper = sum((_dyadic_fraction(value.upper) for value in contributions), Fraction())
    return DyadicClosedInterval(
        lower=_round_fraction_outward(lower, precision_bits, toward_positive=False),
        upper=_round_fraction_outward(upper, precision_bits, toward_positive=True),
    )


def _compare_nonnegative_fraction_to_dyadic(
    value: Fraction, dyadic: ExactDyadic
) -> int:
    """Compare a bounded nonnegative Fraction with a nonnegative dyadic safely."""

    if value < 0 or int(dyadic.mantissa) < 0:
        raise AssertionError("comparison operands must be nonnegative")
    mantissa = int(dyadic.mantissa)
    if value == 0 or mantissa == 0:
        return (value > 0) - (mantissa > 0)
    value_top = _floor_log2_abs(value)
    dyadic_top = mantissa.bit_length() - 1 + dyadic.exponent
    if value_top != dyadic_top:
        return (value_top > dyadic_top) - (value_top < dyadic_top)
    if dyadic.exponent >= 0:
        right = value.denominator * (mantissa << dyadic.exponent)
        left = value.numerator
    else:
        left = value.numerator << (-dyadic.exponent)
        right = value.denominator * mantissa
    return (left > right) - (left < right)


def _target_met(enclosure: DyadicClosedInterval, target_width: ExactDyadic) -> bool:
    return (
        _compare_nonnegative_fraction_to_dyadic(
            _enclosure_width(enclosure), target_width
        )
        <= 0
    )


def _split_interval(
    interval: ClosedRationalInterval,
) -> tuple[ClosedRationalInterval, ClosedRationalInterval]:
    midpoint = CanonicalRational.from_fraction(
        (interval.lower.as_fraction() + interval.upper.as_fraction()) / 2
    )
    return (
        ClosedRationalInterval(lower=interval.lower, upper=midpoint),
        ClosedRationalInterval(lower=midpoint, upper=interval.upper),
    )


def _interval_at_path(
    source: ClosedRationalInterval, path: tuple[int, ...]
) -> ClosedRationalInterval:
    interval = source
    for bit in path:
        if _interval_width(interval) <= 0:
            raise _validation_error(
                "a positive-depth leaf cannot descend from a degenerate interval"
            )
        interval = _split_interval(interval)[bit]
    return interval


def _paths_are_complete(paths: tuple[tuple[int, ...], ...]) -> bool:
    if not paths:
        return False
    for left, right in pairwise(paths):
        if right[: len(left)] == left:
            return False
    depth = max(map(len, paths))
    return sum(1 << (depth - len(path)) for path in paths) == 1 << depth


def _bind_domain_failure_to_expression(
    expression: IntervalExpressionNode,
    failure: IntervalExpressionDomainFailure,
) -> None:
    node = expression
    for child_index in failure.node_path:
        if child_index >= len(node.children):
            raise _validation_error(
                "domain-failure node path does not exist in the source expression"
            )
        node = node.children[child_index]
    if node.op != failure.operation:
        raise _validation_error(
            "domain-failure operation does not match the source expression node"
        )
    if failure.operation == "pow" and (node.exponent is None or node.exponent >= 0):
        raise _validation_error(
            "negative-power domain failure requires a negative source exponent"
        )
    if failure.reason == "SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET":
        raise _validation_error(
            "second-jet differentiability evidence is not an integral-domain failure"
        )


def _bind_partition(
    result: DefiniteIntegralEnclosureResult,
    leaves: tuple[DefiniteIntegralLeaf, ...]
    | tuple[DefiniteIntegralConcludedLeaf, ...],
) -> None:
    if len(leaves) > result.max_leaves:
        raise _validation_error(
            "definite-integral result exceeds the requested leaf budget"
        )
    paths = tuple(leaf.path for leaf in leaves)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise _validation_error(
            "definite-integral leaves must have unique lexicographically ordered paths"
        )
    if any(len(path) > result.max_leaves - 1 for path in paths):
        raise _validation_error(
            "definite-integral leaf exceeds the requested partition-depth bound"
        )
    if not _paths_are_complete(paths):
        raise _validation_error(
            "definite-integral leaf paths must be a complete prefix-free cover"
        )

    source = result.box.intervals[0]
    source_is_degenerate = _interval_width(source) == 0
    zero_leaves = tuple(
        leaf for leaf in leaves if isinstance(leaf, DefiniteIntegralZeroMeasureLeaf)
    )
    if zero_leaves:
        if not source_is_degenerate or len(leaves) != 1 or paths != ((),):
            raise _validation_error(
                "ZERO_MEASURE is reserved for the unique degenerate source leaf"
            )
        zero = zero_leaves[0].contribution
        if _dyadic_fraction(zero.lower) != 0 or _dyadic_fraction(zero.upper) != 0:
            raise _validation_error(
                "the zero-measure contribution must be exactly zero"
            )
    elif source_is_degenerate:
        raise _validation_error(
            "a degenerate source interval requires its exact zero-measure leaf"
        )

    for leaf in leaves:
        interval = _interval_at_path(source, leaf.path)
        if isinstance(leaf, DefiniteIntegralEnclosedLeaf):
            if _interval_width(interval) <= 0:
                raise _validation_error(
                    "an enclosed integral leaf must have positive source measure"
                )
            expected = _leaf_contribution(
                interval, leaf.range_enclosure, result.precision_bits
            )
            if leaf.contribution != expected:
                raise _validation_error(
                    "leaf contribution must be the exact outward dyadic product "
                    "of source width and range enclosure"
                )
        elif isinstance(leaf, DefiniteIntegralDomainUnprovenLeaf):
            if _interval_width(interval) <= 0:
                raise _validation_error(
                    "a domain-unproven integral leaf must have positive source measure"
                )
            _bind_domain_failure_to_expression(result.expression, leaf.domain_failure)


class DefiniteIntegralEnclosureResult(DefiniteIntegralEnclosureRequest):
    """A source-bound integral enclosure or a complete domain-unproven cover."""

    outcome: DefiniteIntegralOutcome

    @model_validator(mode="after")
    def bind_partition_contributions_and_sum(self) -> Self:
        leaves = self.outcome.leaves
        _bind_partition(self, leaves)
        unproven = tuple(
            leaf
            for leaf in leaves
            if isinstance(leaf, DefiniteIntegralDomainUnprovenLeaf)
        )
        if isinstance(self.outcome, DefiniteIntegralDomainUnproven):
            if not unproven:
                raise _validation_error(
                    "DOMAIN_UNPROVEN requires at least one unproven final leaf"
                )
            if len(leaves) != self.max_leaves:
                raise _validation_error(
                    "DOMAIN_UNPROVEN requires exhausting the requested leaf cover"
                )
            return self

        if unproven:
            raise _validation_error(
                "a domain-unproven leaf requires DOMAIN_UNPROVEN outcome"
            )
        contributions = tuple(
            leaf.contribution
            for leaf in leaves
            if isinstance(
                leaf,
                (DefiniteIntegralEnclosedLeaf, DefiniteIntegralZeroMeasureLeaf),
            )
        )
        expected = _summed_enclosure(contributions, self.precision_bits)
        if self.outcome.enclosure != expected:
            raise _validation_error(
                "global integral enclosure must be the outward-rounded exact sum "
                "of every retained leaf contribution"
            )
        target_met = _target_met(expected, self.target_width)
        if target_met != isinstance(self.outcome, DefiniteIntegralTargetMet):
            raise _validation_error(
                "definite-integral outcome must agree with the requested target"
            )
        if (
            isinstance(self.outcome, DefiniteIntegralBudgetExhausted)
            and len(leaves) != self.max_leaves
        ):
            raise _validation_error(
                "BUDGET_EXHAUSTED requires filling the requested leaf budget"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: DefiniteIntegralEnclosureRequest,
        *,
        outcome: DefiniteIntegralOutcome,
    ) -> DefiniteIntegralEnclosureResult:
        """Construct a result after the admitted kernel established every claim."""

        return cls.model_construct(
            expression=request.expression,
            box=request.box,
            precision_bits=request.precision_bits,
            target_width=request.target_width,
            max_leaves=request.max_leaves,
            wall_seconds=request.wall_seconds,
            outcome=outcome,
        )


@dataclass(frozen=True, slots=True)
class _DefiniteIntegralAdmission:
    deadline: float
    root_preflight: _BoxPreflight | None


@dataclass(frozen=True, slots=True)
class _EvaluatedIntegralLeaf:
    path: tuple[int, ...]
    interval: ClosedRationalInterval
    range_enclosure: DyadicClosedInterval | None = None
    contribution: DyadicClosedInterval | None = None
    domain_failure: IntervalExpressionDomainFailure | None = None

    def __post_init__(self) -> None:
        enclosed = self.range_enclosure is not None and self.contribution is not None
        unproven = self.domain_failure is not None
        if enclosed == unproven or (self.range_enclosure is None) != (
            self.contribution is None
        ):
            raise AssertionError("one evaluated leaf must carry exactly one outcome")


def _midpoint_component_digits(box: RationalIntervalBox, depth: int) -> int:
    source_digits = max(
        (
            canonical_rational_component_digits(endpoint)
            for interval in box.intervals
            for endpoint in (interval.lower, interval.upper)
        ),
        default=1,
    )
    return 2 * source_digits + depth + 2


def _estimated_result_bytes(request: DefiniteIntegralEnclosureRequest) -> int:
    source_bytes = len(canonicalize_json(request.model_dump(mode="json")))
    dyadic_interval_bytes = 2 * (MAX_DYADIC_MANTISSA_DIGITS + 64) + 64
    path_bytes = 2 * (request.max_leaves - 1) + 32
    enclosed_leaf_bytes = path_bytes + 2 * dyadic_interval_bytes + 256
    return (
        source_bytes
        + request.max_leaves * enclosed_leaf_bytes
        + dyadic_interval_bytes
        + 8_192
    )


def _require_deadline(deadline: float, stage: str) -> None:
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"definite-integral enclosure deadline expired {stage}"
        )


def _admit_definite_integral(
    request: DefiniteIntegralEnclosureRequest, *, started_at: float
) -> _DefiniteIntegralAdmission:
    deadline = started_at + request.wall_seconds
    bind_request_deadline(deadline)
    _require_deadline(deadline, "before semantic preflight")

    nodes = _bounded_expression_nodes(request.expression)
    maximum_subproblems = 2 * request.max_leaves - 1
    node_work = 2 * len(nodes) * maximum_subproblems
    if node_work > MAX_DEFINITE_INTEGRAL_NODE_WORK:
        raise OperationDomainValidationError(
            location=("max_leaves", "expression"),
            code="analysis.definite_integral.node_work",
            message=(
                f"definite-integral work of {node_work} exact-and-Arb node units "
                f"exceeds the {MAX_DEFINITE_INTEGRAL_NODE_WORK}-unit bound"
            ),
        )
    precision_work = len(nodes) * maximum_subproblems * request.precision_bits
    if precision_work > MAX_DEFINITE_INTEGRAL_PRECISION_WORK:
        raise OperationDomainValidationError(
            location=("precision_bits", "max_leaves", "expression"),
            code="analysis.definite_integral.precision_work",
            message=(
                f"definite-integral precision work of {precision_work} node-bit "
                f"units exceeds the {MAX_DEFINITE_INTEGRAL_PRECISION_WORK}-unit bound"
            ),
        )
    selection_comparisons = request.max_leaves * (request.max_leaves - 1) // 2
    if selection_comparisons > MAX_DEFINITE_INTEGRAL_SELECTION_COMPARISONS:
        raise AssertionError("definite-integral selection accounting is inconsistent")
    summation_units = request.max_leaves * (request.max_leaves + 1)
    if summation_units > MAX_DEFINITE_INTEGRAL_SUMMATION_UNITS:
        raise AssertionError("definite-integral summation accounting is inconsistent")
    accumulator_bits = (
        request.precision_bits
        + 2 * MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT
        + (request.max_leaves - 1).bit_length()
    )
    if accumulator_bits > MAX_DEFINITE_INTEGRAL_ACCUMULATOR_BITS:
        raise OperationDomainValidationError(
            location=("precision_bits", "max_leaves"),
            code="analysis.definite_integral.accumulator_bits",
            message=(
                f"definite-integral exact-sum work of {accumulator_bits} bits "
                f"exceeds the {MAX_DEFINITE_INTEGRAL_ACCUMULATOR_BITS}-bit bound"
            ),
        )
    midpoint_digits = _midpoint_component_digits(request.box, request.max_leaves - 1)
    if midpoint_digits > 2 * MAX_RATIONAL_DIGITS + MAX_DEFINITE_INTEGRAL_DEPTH + 2:
        raise AssertionError("definite-integral midpoint accounting is inconsistent")
    estimated_result_bytes = _estimated_result_bytes(request)
    if estimated_result_bytes > MAX_DEFINITE_INTEGRAL_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("max_leaves",),
            code="analysis.definite_integral.result_bytes",
            message=(
                f"definite-integral result estimate of {estimated_result_bytes} bytes "
                f"exceeds the {MAX_DEFINITE_INTEGRAL_RESULT_BYTES}-byte canonical "
                "output bound"
            ),
        )

    source = request.box.intervals[0]
    root_preflight: _BoxPreflight | None = None
    if _interval_width(source) > 0:
        try:
            root_preflight = _preflight_box_expression(
                request.expression, _rational_box_bounds(request.box)
            )
        except ValueError as exc:
            raise OperationDomainValidationError(
                location=("expression",),
                code="analysis.definite_integral.intermediate_bound",
                message=str(exc),
            ) from exc
    _require_deadline(deadline, "after semantic preflight")
    return _DefiniteIntegralAdmission(
        deadline=deadline,
        root_preflight=root_preflight,
    )


def _leaf_box(
    source: RationalIntervalBox, interval: ClosedRationalInterval
) -> RationalIntervalBox:
    return RationalIntervalBox(variables=source.variables, intervals=(interval,))


def _evaluate_integral_leaf(
    request: DefiniteIntegralEnclosureRequest,
    path: tuple[int, ...],
    interval: ClosedRationalInterval,
    *,
    deadline: float,
    preflight: _BoxPreflight | None = None,
) -> _EvaluatedIntegralLeaf:
    _require_deadline(deadline, "before a leaf domain preflight")
    box = _leaf_box(request.box, interval)
    if preflight is None:
        try:
            preflight = _preflight_box_expression(
                request.expression, _rational_box_bounds(box)
            )
        except ValueError as exc:
            raise OperationDomainValidationError(
                location=("expression",),
                code="analysis.definite_integral.intermediate_bound",
                message=str(exc),
            ) from exc
    _require_deadline(deadline, "after a leaf domain preflight")
    if isinstance(preflight, IntervalExpressionDomainFailure):
        return _EvaluatedIntegralLeaf(
            path=path, interval=interval, domain_failure=preflight
        )
    assert isinstance(preflight, _RationalBounds)

    _require_deadline(deadline, "before an Arb leaf evaluation")
    try:
        from flint import ctx

        with ctx.workprec(request.precision_bits):
            variables = {
                request.box.variables[0]: arb_source_interval(interval),
            }
            result = _evaluate_box_expression(request.expression, variables)
            if isinstance(result, IntervalExpressionDomainFailure):
                raise RuntimeError(
                    "pinned Arb disagreed with the exact admitted leaf domain"
                )
            if isinstance(result, _BoxEvaluationFailure) or not result.is_finite():
                raise RuntimeError(
                    "pinned Arb returned no finite integral-leaf enclosure"
                )
            lower_mantissa, lower_exponent = result.lower().man_exp()
            upper_mantissa, upper_exponent = result.upper().man_exp()
            endpoints = dyadic_endpoints(
                lower_mantissa,
                lower_exponent,
                upper_mantissa,
                upper_exponent,
            )
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(
            "pinned Arb rejected an admitted integral-leaf evaluation"
        ) from exc
    _require_deadline(deadline, "after an Arb leaf evaluation")
    if endpoints is None or any(
        abs(endpoint.exponent) > MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT
        for endpoint in endpoints
    ):
        raise RuntimeError(
            "pinned Arb produced an integral endpoint outside the admitted dyadic "
            "source envelope"
        )
    range_enclosure = DyadicClosedInterval(lower=endpoints[0], upper=endpoints[1])
    leaf = _EvaluatedIntegralLeaf(
        path=path,
        interval=interval,
        range_enclosure=range_enclosure,
        contribution=_leaf_contribution(
            interval, range_enclosure, request.precision_bits
        ),
    )
    _require_deadline(deadline, "after leaf contribution construction")
    return leaf


def _public_leaves(
    leaves: tuple[_EvaluatedIntegralLeaf, ...],
) -> tuple[DefiniteIntegralLeaf, ...]:
    public: list[DefiniteIntegralLeaf] = []
    for leaf in sorted(leaves, key=lambda value: value.path):
        if leaf.domain_failure is not None:
            public.append(
                DefiniteIntegralDomainUnprovenLeaf(
                    path=leaf.path, domain_failure=leaf.domain_failure
                )
            )
            continue
        assert leaf.range_enclosure is not None and leaf.contribution is not None
        public.append(
            DefiniteIntegralEnclosedLeaf(
                path=leaf.path,
                range_enclosure=leaf.range_enclosure,
                contribution=leaf.contribution,
            )
        )
    return tuple(public)


def _finish_result(
    request: DefiniteIntegralEnclosureRequest,
    public_leaves: tuple[DefiniteIntegralLeaf, ...],
    *,
    enclosure: DyadicClosedInterval | None,
    deadline: float,
) -> DefiniteIntegralEnclosureResult:
    if any(
        isinstance(leaf, DefiniteIntegralDomainUnprovenLeaf) for leaf in public_leaves
    ):
        assert enclosure is None
        result = DefiniteIntegralEnclosureResult._from_kernel(
            request,
            outcome=DefiniteIntegralDomainUnproven(leaves=public_leaves),
        )
    else:
        assert enclosure is not None
        concluded_leaves = tuple(
            leaf
            for leaf in public_leaves
            if isinstance(
                leaf,
                (DefiniteIntegralEnclosedLeaf, DefiniteIntegralZeroMeasureLeaf),
            )
        )
        assert len(concluded_leaves) == len(public_leaves)
        outcome: DefiniteIntegralTargetMet | DefiniteIntegralBudgetExhausted
        if _target_met(enclosure, request.target_width):
            outcome = DefiniteIntegralTargetMet(
                enclosure=enclosure, leaves=concluded_leaves
            )
        else:
            outcome = DefiniteIntegralBudgetExhausted(
                enclosure=enclosure, leaves=concluded_leaves
            )
        result = DefiniteIntegralEnclosureResult._from_kernel(
            request,
            outcome=outcome,
        )
    _require_deadline(deadline, "after result construction")
    return result


def _finish_zero_measure_result(
    request: DefiniteIntegralEnclosureRequest, *, deadline: float
) -> DefiniteIntegralEnclosureResult:
    zero = DyadicClosedInterval(
        lower=ExactDyadic(mantissa="0", exponent=0),
        upper=ExactDyadic(mantissa="0", exponent=0),
    )
    result = DefiniteIntegralEnclosureResult._from_kernel(
        request,
        outcome=DefiniteIntegralTargetMet(
            enclosure=zero,
            leaves=(DefiniteIntegralZeroMeasureLeaf(path=(), contribution=zero),),
        ),
    )
    _require_deadline(deadline, "after result construction")
    return result


def _unrounded_contribution_width(leaf: _EvaluatedIntegralLeaf) -> Fraction:
    assert leaf.range_enclosure is not None
    return _interval_width(leaf.interval) * _enclosure_width(leaf.range_enclosure)


def _select_leaf(leaves: tuple[_EvaluatedIntegralLeaf, ...]) -> _EvaluatedIntegralLeaf:
    unproven = tuple(leaf for leaf in leaves if leaf.domain_failure is not None)
    if unproven:
        return min(
            unproven,
            key=lambda leaf: (-_interval_width(leaf.interval), leaf.path),
        )
    return min(
        leaves,
        key=lambda leaf: (-_unrounded_contribution_width(leaf), leaf.path),
    )


def _compute_definite_integral_enclosure(
    request: DefiniteIntegralEnclosureRequest,
    *,
    native_started_at: float | None = None,
) -> DefiniteIntegralEnclosureResult:
    execution = current_request_execution()
    started_at = (
        execution.started_at
        if execution is not None
        else native_started_at
        if native_started_at is not None
        else monotonic()
    )
    admission = _admit_definite_integral(request, started_at=started_at)
    source = request.box.intervals[0]
    if _interval_width(source) == 0:
        return _finish_zero_measure_result(request, deadline=admission.deadline)

    leaves: tuple[_EvaluatedIntegralLeaf, ...] = (
        _evaluate_integral_leaf(
            request,
            (),
            source,
            deadline=admission.deadline,
            preflight=admission.root_preflight,
        ),
    )
    while True:
        _require_deadline(admission.deadline, "before partition refinement")
        has_unproven = any(leaf.domain_failure is not None for leaf in leaves)
        public: tuple[DefiniteIntegralLeaf, ...] | None = None
        enclosure: DyadicClosedInterval | None = None
        if not has_unproven:
            public = _public_leaves(leaves)
            _require_deadline(admission.deadline, "after leaf result construction")
            contributions = tuple(
                leaf.contribution
                for leaf in public
                if isinstance(leaf, DefiniteIntegralEnclosedLeaf)
            )
            enclosure = _summed_enclosure(contributions, request.precision_bits)
            _require_deadline(admission.deadline, "after exact leaf summation")
            if _target_met(enclosure, request.target_width):
                return _finish_result(
                    request,
                    public,
                    enclosure=enclosure,
                    deadline=admission.deadline,
                )
        if len(leaves) >= request.max_leaves:
            if public is None:
                public = _public_leaves(leaves)
            return _finish_result(
                request,
                public,
                enclosure=enclosure,
                deadline=admission.deadline,
            )

        selected = _select_leaf(leaves)
        child_intervals = _split_interval(selected.interval)
        lower = _evaluate_integral_leaf(
            request,
            (*selected.path, 0),
            child_intervals[0],
            deadline=admission.deadline,
        )
        _require_deadline(admission.deadline, "between child evaluations")
        upper = _evaluate_integral_leaf(
            request,
            (*selected.path, 1),
            child_intervals[1],
            deadline=admission.deadline,
        )
        leaves = (
            *(leaf for leaf in leaves if leaf.path != selected.path),
            lower,
            upper,
        )
        leaves = tuple(sorted(leaves, key=lambda leaf: leaf.path))


def definite_integral_enclosure(
    expression: IntervalExpressionNode,
    box: RationalIntervalBox,
    target_width: ExactDyadic,
    *,
    precision_bits: int = 128,
    max_leaves: int = 32,
    wall_seconds: int = 30,
) -> DefiniteIntegralEnclosureResult:
    """Return a rigorous enclosure of one definite integral."""

    started_at = monotonic()
    request = DefiniteIntegralEnclosureRequest(
        expression=expression,
        box=box,
        precision_bits=precision_bits,
        target_width=target_width,
        max_leaves=max_leaves,
        wall_seconds=wall_seconds,
    )
    return _compute_definite_integral_enclosure(request, native_started_at=started_at)


DEFINITE_INTEGRAL_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.definite_integral_enclosure.compute",
        title="Enclose the definite integral of an elementary expression",
        description=(
            "Return an outward-rounded dyadic enclosure of one definite integral "
            "over an exact rational interval. Each leaf uses pinned Arb natural "
            "interval arithmetic; exact rational leaf width times its range is "
            "rounded outward and all contributions are summed exactly before one "
            "final outward conversion. Binary midpoint paths identify the complete "
            "deterministic cover. DOMAIN_UNPROVEN means interval dependency left at "
            "least one real-domain obligation unproved and carries no integral "
            "conclusion. BUDGET_EXHAUSTED retains a sound enclosure after filling "
            "the requested leaf budget. The envelope admits at most "
            f"{MAX_DEFINITE_INTEGRAL_LEAVES:,} leaves, "
            f"{MAX_DEFINITE_INTEGRAL_SUBPROBLEMS:,} subproblems, "
            f"{MAX_DEFINITE_INTEGRAL_NODE_WORK:,} exact-and-Arb node units, "
            f"{MAX_DEFINITE_INTEGRAL_PRECISION_WORK:,} precision-weighted node-bit "
            "units, "
            f"{MAX_DEFINITE_INTEGRAL_SELECTION_COMPARISONS:,} deterministic "
            "leaf-selection comparisons, "
            f"{MAX_DEFINITE_INTEGRAL_SUMMATION_UNITS:,} exact dyadic endpoint-add "
            "units, 4,096 Arb precision bits, and one complete result within the "
            "10 MiB canonical output bound."
        ),
        request_type=DefiniteIntegralEnclosureRequest,
        result_type=DefiniteIntegralEnclosureResult,
        run=_compute_definite_integral_enclosure,
        tags=(
            "analysis",
            "interval",
            "expression",
            "integral",
            "definite-integral",
            "adaptive",
            "partition",
            "arb",
            "validated",
            "bounded",
        ),
        examples=(
            example(
                "linear_unit_interval",
                (
                    "Enclose the integral of t over 0 <= t <= 1 until width 1/4; "
                    "the expression and one-axis box must name the same variable."
                ),
                {
                    "expression": {"op": "var", "variable": "t"},
                    "box": {
                        "variables": ["t"],
                        "intervals": [
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "precision_bits": 128,
                    "target_width": {"mantissa": "1", "exponent": -2},
                    "max_leaves": 8,
                    "wall_seconds": 30,
                },
            ),
        ),
    ),
)


__all__ = [
    "DEFINITE_INTEGRAL_ENCLOSURE_OPERATIONS",
    "MAX_DEFINITE_INTEGRAL_ACCUMULATOR_BITS",
    "MAX_DEFINITE_INTEGRAL_DEPTH",
    "MAX_DEFINITE_INTEGRAL_DYADIC_EXPONENT",
    "MAX_DEFINITE_INTEGRAL_LEAVES",
    "MAX_DEFINITE_INTEGRAL_NODE_WORK",
    "MAX_DEFINITE_INTEGRAL_PRECISION_WORK",
    "MAX_DEFINITE_INTEGRAL_RESULT_BYTES",
    "MAX_DEFINITE_INTEGRAL_SELECTION_COMPARISONS",
    "MAX_DEFINITE_INTEGRAL_SUBPROBLEMS",
    "MAX_DEFINITE_INTEGRAL_SUMMATION_UNITS",
    "MAX_DEFINITE_INTEGRAL_WALL_SECONDS",
    "DefiniteIntegralBudgetExhausted",
    "DefiniteIntegralDomainUnproven",
    "DefiniteIntegralDomainUnprovenLeaf",
    "DefiniteIntegralEnclosedLeaf",
    "DefiniteIntegralEnclosureRequest",
    "DefiniteIntegralEnclosureResult",
    "DefiniteIntegralTargetMet",
    "DefiniteIntegralZeroMeasureLeaf",
    "definite_integral_enclosure",
]
