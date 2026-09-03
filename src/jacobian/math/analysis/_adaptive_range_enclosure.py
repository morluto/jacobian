"""Adaptive complete range enclosures over exact rational box partitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fractions import Fraction
from time import monotonic
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian._flint import flint_workprec
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._box_enclosure import (
    _BoxEvaluationFailure,
    _evaluate_box_expression,
    _preflight_box_expression,
)
from jacobian.math.analysis._models import (
    MAX_BOX_PREFLIGHT_TEMPORARY_BITS,
    MAX_RATIONAL_BOX_ENDPOINT_DIGITS,
    MAX_RATIONAL_BOX_PARTITION_DEPTH,
    MAX_RATIONAL_DIGITS,
    DyadicClosedInterval,
    ExactDyadic,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    RationalIntervalBox,
    _bound_raw_box,
    _bound_raw_rational,
    _bounded_expression_nodes,
    _IntervalExpressionBoxRequest,
    _rational_box_bounds,
    _validation_error,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval

MAX_ADAPTIVE_RANGE_LEAVES = 1_024
MAX_ADAPTIVE_RANGE_DEPTH = MAX_RATIONAL_BOX_PARTITION_DEPTH
MAX_ADAPTIVE_RANGE_EVALUATIONS = 4_096
MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS = 131_072
MAX_ADAPTIVE_RANGE_PRECISION_WORK = 268_435_456
MAX_ADAPTIVE_RANGE_WALL_SECONDS = 120

# Admission bounds every exact source-box value by an 8,192-bit rational and
# every Arb work precision by 4,096 bits. An outward endpoint can therefore
# require at most their sum in its binary exponent. The existing temporary
# Fraction ceiling is a slightly larger, named bound that also keeps public
# result validation from expanding an enormous authored dyadic exponent.
MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT = MAX_BOX_PREFLIGHT_TEMPORARY_BITS

type AdaptiveRangeBudgetReason = Literal[
    "MAX_LEAVES", "MAX_DEPTH", "MAX_EVALUATIONS", "MAX_PRECISION"
]


def _adaptive_source_endpoint_digit_cap(max_depth: int) -> int:
    if max_depth == 0:
        return MAX_RATIONAL_BOX_ENDPOINT_DIGITS
    return (MAX_RATIONAL_BOX_ENDPOINT_DIGITS - max_depth - 2) // 2


class AdaptiveRangeTargetMet(StrictModel):
    """The complete enclosure is no wider than the requested target."""

    status: Literal["TARGET_MET"] = "TARGET_MET"


class AdaptiveRangeBudgetExhausted(StrictModel):
    """The complete enclosure is sound but the finite schedule stopped first."""

    status: Literal["BUDGET_EXHAUSTED"] = "BUDGET_EXHAUSTED"
    reason: AdaptiveRangeBudgetReason


class AdaptiveRangeDomainUnproven(StrictModel):
    """No global range is claimed because one final Arb leaf is uncertain."""

    status: Literal["DOMAIN_UNPROVEN"] = "DOMAIN_UNPROVEN"
    reason: AdaptiveRangeBudgetReason = Field(
        description=(
            "The deterministic finite refinement resource that prevented every "
            "uncertain leaf from obtaining an Arb enclosure."
        )
    )


type AdaptiveRangeDisposition = Annotated[
    AdaptiveRangeTargetMet | AdaptiveRangeBudgetExhausted | AdaptiveRangeDomainUnproven,
    Field(discriminator="status"),
]

type _AdaptiveRangeConcludedDisposition = Annotated[
    AdaptiveRangeTargetMet | AdaptiveRangeBudgetExhausted,
    Field(discriminator="status"),
]


class AdaptiveRangeEnclosureRequest(_IntervalExpressionBoxRequest):
    """Bound one complete elementary-expression range by adaptive bisection."""

    box: RationalIntervalBox = Field(
        description=(
            "Complete source box. With max_depth=0, endpoint components admit at "
            f"most {MAX_RATIONAL_BOX_ENDPOINT_DIGITS} decimal digits. For positive "
            "greatest reachable leaf depth d after all refinement caps, the source "
            "cap is "
            f"floor(({MAX_RATIONAL_BOX_ENDPOINT_DIGITS} - d - 2) / 2), so every "
            "derived midpoint remains inside the shared rational-box envelope."
        )
    )
    target_width: CanonicalRational = Field(
        description=(
            "A positive exact rational target for upper-lower. Components admit "
            "at most 128 decimal digits."
        )
    )
    maximum_precision_bits: StrictInt = Field(
        default=512,
        ge=32,
        le=4096,
        description=(
            "Final precision in the deterministic doubling schedule; it must be "
            "at least precision_bits."
        ),
    )
    max_leaves: StrictInt = Field(
        default=32,
        ge=1,
        le=MAX_ADAPTIVE_RANGE_LEAVES,
        description="Maximum retained leaves in the complete binary partition.",
    )
    max_depth: StrictInt = Field(
        default=8,
        ge=0,
        le=MAX_ADAPTIVE_RANGE_DEPTH,
        description="Maximum number of exact midpoint bisections on one leaf path.",
    )
    max_evaluations: StrictInt = Field(
        default=128,
        ge=1,
        le=MAX_ADAPTIVE_RANGE_EVALUATIONS,
        description=(
            "Maximum Arb expression evaluations. A split is attempted only when "
            "two child evaluations remain."
        ),
    )
    wall_seconds: StrictInt = Field(
        default=30,
        ge=1,
        le=MAX_ADAPTIVE_RANGE_WALL_SECONDS,
        description=(
            "Shared owner deadline in seconds, measured from dispatch start (or "
            "native-function entry) through final result construction."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_adaptive_fields(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if isinstance(value, Mapping):
            _bound_raw_rational(value.get("target_width"), "adaptive target width")
            if cls is AdaptiveRangeEnclosureRequest:
                _bound_raw_box(
                    value.get("box"),
                    max_digits=_adaptive_source_endpoint_digit_cap(
                        _raw_maximum_leaf_depth(value)
                    ),
                )
        return value

    @model_validator(mode="after")
    def require_precision_schedule(self) -> Self:
        if self.maximum_precision_bits < self.precision_bits:
            raise _validation_error(
                "maximum_precision_bits must be at least precision_bits"
            )
        return self


@dataclass(frozen=True, slots=True)
class _AdaptiveRangeProblem:
    """Canonical domain input shared by native and wire execution paths."""

    expression: IntervalExpressionNode
    box: RationalIntervalBox
    target_width: CanonicalRational
    precision_bits: int
    maximum_precision_bits: int
    max_leaves: int
    max_depth: int
    max_evaluations: int
    wall_seconds: int


def _problem_from_request(
    request: AdaptiveRangeEnclosureRequest,
) -> _AdaptiveRangeProblem:
    return _AdaptiveRangeProblem(
        expression=request.expression,
        box=request.box,
        target_width=request.target_width,
        precision_bits=request.precision_bits,
        maximum_precision_bits=request.maximum_precision_bits,
        max_leaves=request.max_leaves,
        max_depth=request.max_depth,
        max_evaluations=request.max_evaluations,
        wall_seconds=request.wall_seconds,
    )


class _AdaptiveRangeLeafPath(StrictModel):
    """One canonically addressed leaf in the exact midpoint partition."""

    path: tuple[StrictInt, ...] = Field(
        max_length=MAX_ADAPTIVE_RANGE_DEPTH,
        description=(
            "Binary child choices from the source box. At every step 0 selects "
            "the lower and 1 the upper exact-midpoint child of the widest positive "
            "coordinate, with source-axis ties."
        ),
    )
    box: RationalIntervalBox = Field(
        description=(
            "The exact source-bound midpoint subbox. Every serialized endpoint "
            "numerator and denominator has at most "
            f"{MAX_RATIONAL_BOX_ENDPOINT_DIGITS} decimal digits."
        )
    )

    @field_validator("path", mode="before")
    @classmethod
    def preserve_bounded_path(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_ADAPTIVE_RANGE_DEPTH:
            raise _validation_error(
                "adaptive leaf path exceeds the bisection-depth bound"
            )
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_binary_path(self) -> Self:
        if any(bit not in (0, 1) for bit in self.path):
            raise _validation_error("adaptive leaf paths contain only 0 and 1")
        return self


class AdaptiveRangeLeaf(_AdaptiveRangeLeafPath):
    """One leaf with a sound expression enclosure."""

    status: Literal["ENCLOSED"] = "ENCLOSED"
    enclosure: DyadicClosedInterval


class AdaptiveRangeDomainUnprovenLeaf(_AdaptiveRangeLeafPath):
    """One leaf whose fixed-precision Arb evaluation remained uncertain."""

    status: Literal["DOMAIN_UNPROVEN"] = "DOMAIN_UNPROVEN"
    domain_failure: IntervalExpressionDomainFailure


type AdaptiveRangeFinalLeaf = Annotated[
    AdaptiveRangeLeaf | AdaptiveRangeDomainUnprovenLeaf,
    Field(discriminator="status"),
]


def _precision_schedule(initial: int, maximum: int) -> tuple[int, ...]:
    schedule = [initial]
    while schedule[-1] < maximum:
        schedule.append(min(2 * schedule[-1], maximum))
    return tuple(schedule)


def _planned_schedule_and_splits(
    *,
    precision_bits: int,
    maximum_precision_bits: int,
    max_leaves: int,
    max_depth: int,
    max_evaluations: int,
    splittable: bool,
) -> tuple[tuple[int, ...], int, int]:
    schedule = _precision_schedule(precision_bits, maximum_precision_bits)
    root_evaluations = min(max_evaluations, len(schedule))
    planned_splits = 0
    if root_evaluations == len(schedule) and splittable:
        planned_splits = min(
            max_leaves - 1,
            (1 << max_depth) - 1,
            (max_evaluations - root_evaluations) // 2,
        )
    return schedule, root_evaluations, planned_splits


def _raw_maximum_leaf_depth(value: Mapping[object, object]) -> int:
    precision_bits = value.get("precision_bits", 128)
    maximum_precision_bits = value.get("maximum_precision_bits", 512)
    max_leaves = value.get("max_leaves", 32)
    max_depth = value.get("max_depth", 8)
    max_evaluations = value.get("max_evaluations", 128)
    if not (
        type(precision_bits) is int
        and 32 <= precision_bits <= 4_096
        and type(maximum_precision_bits) is int
        and precision_bits <= maximum_precision_bits <= 4_096
        and type(max_leaves) is int
        and 1 <= max_leaves <= MAX_ADAPTIVE_RANGE_LEAVES
        and type(max_depth) is int
        and 0 <= max_depth <= MAX_ADAPTIVE_RANGE_DEPTH
        and type(max_evaluations) is int
        and 1 <= max_evaluations <= MAX_ADAPTIVE_RANGE_EVALUATIONS
    ):
        return MAX_ADAPTIVE_RANGE_DEPTH
    _, _, split_count = _planned_schedule_and_splits(
        precision_bits=precision_bits,
        maximum_precision_bits=maximum_precision_bits,
        max_leaves=max_leaves,
        max_depth=max_depth,
        max_evaluations=max_evaluations,
        splittable=_raw_box_might_split(value.get("box")),
    )
    return min(max_depth, split_count)


def _raw_box_might_split(box: object) -> bool:
    if isinstance(box, RationalIntervalBox):
        return _widest_coordinate(box) is not None
    if not isinstance(box, Mapping):
        return True
    intervals = box.get("intervals")
    if not isinstance(intervals, (list, tuple)):
        return True
    for interval in intervals:
        if isinstance(interval, ClosedRationalInterval):
            if interval.lower != interval.upper:
                return True
            continue
        if not isinstance(interval, Mapping):
            return True
        lower = interval.get("lower")
        upper = interval.get("upper")
        if lower != upper:
            return True
    return False


def _dyadic_fraction(value: ExactDyadic) -> Fraction:
    _require_adaptive_dyadic_endpoint(value)
    return value.as_fraction()


def _require_adaptive_dyadic_endpoint(value: ExactDyadic) -> None:
    if abs(value.exponent) > MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT:
        raise _validation_error(
            "adaptive dyadic exponent exceeds the admitted source-and-precision bound"
        )


def _require_adaptive_dyadic_interval(enclosure: DyadicClosedInterval) -> None:
    _require_adaptive_dyadic_endpoint(enclosure.lower)
    _require_adaptive_dyadic_endpoint(enclosure.upper)


def _enclosure_width(enclosure: DyadicClosedInterval) -> Fraction:
    return _dyadic_fraction(enclosure.upper) - _dyadic_fraction(enclosure.lower)


def _enclosure_hull(
    enclosures: tuple[DyadicClosedInterval, ...],
) -> DyadicClosedInterval:
    lower = min(
        (enclosure.lower for enclosure in enclosures),
        key=_dyadic_fraction,
    )
    upper = max(
        (enclosure.upper for enclosure in enclosures),
        key=_dyadic_fraction,
    )
    return DyadicClosedInterval(lower=lower, upper=upper)


def _widest_coordinate(box: RationalIntervalBox) -> int | None:
    widest_index: int | None = None
    widest_width = Fraction(0)
    for index, interval in enumerate(box.intervals):
        width = interval.upper.as_fraction() - interval.lower.as_fraction()
        if width > widest_width:
            widest_index = index
            widest_width = width
    return widest_index


def _split_box(
    box: RationalIntervalBox, coordinate: int
) -> tuple[RationalIntervalBox, RationalIntervalBox]:
    selected = box.intervals[coordinate]
    midpoint = CanonicalRational.from_fraction(
        (selected.lower.as_fraction() + selected.upper.as_fraction()) / 2
    )
    lower_intervals = list(box.intervals)
    upper_intervals = list(box.intervals)
    lower_intervals[coordinate] = ClosedRationalInterval(
        lower=selected.lower, upper=midpoint
    )
    upper_intervals[coordinate] = ClosedRationalInterval(
        lower=midpoint, upper=selected.upper
    )
    return (
        RationalIntervalBox(variables=box.variables, intervals=tuple(lower_intervals)),
        RationalIntervalBox(variables=box.variables, intervals=tuple(upper_intervals)),
    )


def _paths_are_complete(paths: tuple[tuple[int, ...], ...]) -> bool:
    for index, path in enumerate(paths):
        if any(other[: len(path)] == path for other in paths[index + 1 :]):
            return False
    depth = max(map(len, paths), default=0)
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
            "second-jet differentiability evidence is not a range-domain failure"
        )


def _bind_leaf_partition(result: AdaptiveRangeEnclosureResult) -> None:
    if len(result.leaves) > result.max_leaves:
        raise _validation_error("adaptive result exceeds the requested leaf budget")
    paths = tuple(leaf.path for leaf in result.leaves)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise _validation_error(
            "adaptive leaves must have unique lexicographically ordered paths"
        )
    if any(len(path) > result.max_depth for path in paths):
        raise _validation_error("adaptive leaf exceeds the requested depth budget")
    if not _paths_are_complete(paths):
        raise _validation_error(
            "adaptive leaf paths must be a complete prefix-free binary cover"
        )
    for leaf in result.leaves:
        if isinstance(leaf, AdaptiveRangeDomainUnprovenLeaf):
            _bind_domain_failure_to_expression(result.expression, leaf.domain_failure)
        else:
            _require_adaptive_dyadic_interval(leaf.enclosure)


class _AdaptiveRangeResultSchemaBase(AdaptiveRangeEnclosureRequest):
    """Fields shared by the two schema-visible result branches."""

    evaluations_used: StrictInt = Field(ge=1, le=MAX_ADAPTIVE_RANGE_EVALUATIONS)
    maximum_precision_bits_used: StrictInt = Field(ge=32, le=4096)


class AdaptiveRangeConcludedResult(_AdaptiveRangeResultSchemaBase):
    """Schema branch for a global enclosure conclusion."""

    enclosure: DyadicClosedInterval
    disposition: _AdaptiveRangeConcludedDisposition
    leaves: tuple[AdaptiveRangeLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_ADAPTIVE_RANGE_LEAVES,
    )


class AdaptiveRangeDomainUnprovenResult(_AdaptiveRangeResultSchemaBase):
    """Schema branch for a complete cover with a local domain uncertainty."""

    enclosure: None
    disposition: AdaptiveRangeDomainUnproven
    leaves: tuple[AdaptiveRangeFinalLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_ADAPTIVE_RANGE_LEAVES,
        json_schema_extra={
            "contains": {
                "properties": {"status": {"const": "DOMAIN_UNPROVEN"}},
                "required": ["status"],
            },
            "minContains": 1,
        },
    )


class AdaptiveRangeEnclosureResult(AdaptiveRangeEnclosureRequest):
    """A complete source-bound range enclosure or typed finite nonconclusion."""

    enclosure: DyadicClosedInterval | None = Field(
        description=(
            "The hull of every retained leaf enclosure; absent exactly when the "
            "disposition is DOMAIN_UNPROVEN."
        ),
    )
    disposition: AdaptiveRangeDisposition
    leaves: tuple[AdaptiveRangeFinalLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_ADAPTIVE_RANGE_LEAVES,
        description=(
            "Canonical path-ordered complete partition. Leaf interiors are "
            "disjoint; adjacent closed leaves share their bisection face."
        ),
    )
    evaluations_used: StrictInt = Field(ge=1, le=MAX_ADAPTIVE_RANGE_EVALUATIONS)
    maximum_precision_bits_used: StrictInt = Field(ge=32, le=4096)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> JsonSchemaValue:
        """Publish mutually exclusive concluded and domain-uncertain shapes."""

        return {
            "oneOf": [
                handler(AdaptiveRangeConcludedResult.__pydantic_core_schema__),
                handler(AdaptiveRangeDomainUnprovenResult.__pydantic_core_schema__),
            ],
            "title": cls.__name__,
            "description": cls.__doc__,
        }

    @field_validator("leaves", mode="before")
    @classmethod
    def preserve_bounded_leaves(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_ADAPTIVE_RANGE_LEAVES:
            raise _validation_error("adaptive result exceeds its leaf bound")
        if isinstance(value, (list, tuple)):
            for leaf in value:
                if not isinstance(leaf, Mapping):
                    continue
                path = leaf.get("path")
                if (
                    isinstance(path, (list, tuple))
                    and len(path) > MAX_ADAPTIVE_RANGE_DEPTH
                ):
                    raise _validation_error(
                        "adaptive leaf path exceeds the bisection-depth bound"
                    )
                _bound_raw_box(leaf.get("box"))
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bind_partition_and_disposition_state(self) -> Self:
        _bind_leaf_partition(self)
        if self.enclosure is not None:
            _require_adaptive_dyadic_interval(self.enclosure)
        if self.evaluations_used > self.max_evaluations:
            raise _validation_error(
                "adaptive result exceeds the requested evaluation budget"
            )
        if not (
            self.precision_bits
            <= self.maximum_precision_bits_used
            <= self.maximum_precision_bits
        ):
            raise _validation_error(
                "adaptive used precision must lie in the requested precision range"
            )
        unproven = tuple(
            leaf
            for leaf in self.leaves
            if isinstance(leaf, AdaptiveRangeDomainUnprovenLeaf)
        )
        if isinstance(self.disposition, AdaptiveRangeDomainUnproven):
            if not unproven:
                raise _validation_error(
                    "DOMAIN_UNPROVEN requires at least one uncertain final leaf"
                )
            if self.enclosure is not None:
                raise _validation_error(
                    "DOMAIN_UNPROVEN cannot carry a global enclosure"
                )
            return self
        if unproven:
            raise _validation_error(
                "an uncertain adaptive leaf requires DOMAIN_UNPROVEN disposition"
            )
        if self.enclosure is None:
            raise _validation_error(
                "a concluded adaptive disposition requires a global enclosure"
            )
        enclosure_width = (
            self.enclosure.upper.as_fraction() - self.enclosure.lower.as_fraction()
        )
        target_width = self.target_width.as_fraction()
        if isinstance(self.disposition, AdaptiveRangeTargetMet):
            if enclosure_width > target_width:
                raise _validation_error(
                    "TARGET_MET requires enclosure width at most target_width"
                )
        elif enclosure_width <= target_width:
            raise _validation_error(
                "BUDGET_EXHAUSTED requires enclosure width above target_width"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        problem: _AdaptiveRangeProblem,
        *,
        enclosure: DyadicClosedInterval | None,
        disposition: AdaptiveRangeDisposition,
        leaves: tuple[AdaptiveRangeFinalLeaf, ...],
        evaluations_used: int,
        maximum_precision_bits_used: int,
    ) -> AdaptiveRangeEnclosureResult:
        """Construct a result after the kernel established every range claim."""

        return cls.model_construct(
            expression=problem.expression,
            box=problem.box,
            precision_bits=problem.precision_bits,
            target_width=problem.target_width,
            maximum_precision_bits=problem.maximum_precision_bits,
            max_leaves=problem.max_leaves,
            max_depth=problem.max_depth,
            max_evaluations=problem.max_evaluations,
            wall_seconds=problem.wall_seconds,
            enclosure=enclosure,
            disposition=disposition,
            leaves=leaves,
            evaluations_used=evaluations_used,
            maximum_precision_bits_used=maximum_precision_bits_used,
        )


@dataclass(frozen=True, slots=True)
class _AdaptiveRangeExecutionPlan:
    precision_schedule: tuple[int, ...]
    planned_splits: int
    planned_evaluations: int
    planned_leaf_count: int
    planned_maximum_leaf_depth: int
    precision_bits_per_expression_node: int


@dataclass(frozen=True, slots=True)
class _AdaptiveRangeAdmission:
    plan: _AdaptiveRangeExecutionPlan
    planned_node_evaluations: int
    target_width: Fraction
    deadline: float

    @property
    def precision_schedule(self) -> tuple[int, ...]:
        return self.plan.precision_schedule

    @property
    def planned_evaluations(self) -> int:
        return self.plan.planned_evaluations


@dataclass(frozen=True, slots=True)
class _EvaluatedAdaptiveRangeLeaf:
    path: tuple[int, ...]
    box: RationalIntervalBox
    enclosure: DyadicClosedInterval | None = None
    domain_failure: IntervalExpressionDomainFailure | None = None
    split_coordinate: int | None = dataclass_field(init=False, repr=False)
    enclosure_lower: Fraction | None = dataclass_field(init=False, repr=False)
    enclosure_upper: Fraction | None = dataclass_field(init=False, repr=False)
    enclosure_width: Fraction | None = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.enclosure is None) == (self.domain_failure is None):
            raise AssertionError("one adaptive leaf must carry exactly one outcome")
        object.__setattr__(self, "split_coordinate", _widest_coordinate(self.box))
        if self.enclosure is None:
            lower = upper = width = None
        else:
            lower = _dyadic_fraction(self.enclosure.lower)
            upper = _dyadic_fraction(self.enclosure.upper)
            width = upper - lower
        object.__setattr__(self, "enclosure_lower", lower)
        object.__setattr__(self, "enclosure_upper", upper)
        object.__setattr__(self, "enclosure_width", width)


def _plan_adaptive_range(problem: _AdaptiveRangeProblem) -> _AdaptiveRangeExecutionPlan:
    schedule, root_evaluations, planned_splits = _planned_schedule_and_splits(
        precision_bits=problem.precision_bits,
        maximum_precision_bits=problem.maximum_precision_bits,
        max_leaves=problem.max_leaves,
        max_depth=problem.max_depth,
        max_evaluations=problem.max_evaluations,
        splittable=_widest_coordinate(problem.box) is not None,
    )
    planned_evaluations = root_evaluations + 2 * planned_splits
    return _AdaptiveRangeExecutionPlan(
        precision_schedule=schedule,
        planned_splits=planned_splits,
        planned_evaluations=planned_evaluations,
        planned_leaf_count=planned_splits + 1,
        planned_maximum_leaf_depth=min(problem.max_depth, planned_splits),
        precision_bits_per_expression_node=(
            sum(schedule[:root_evaluations])
            + 2 * planned_splits * problem.maximum_precision_bits
        ),
    )


def _midpoint_component_digits(box: RationalIntervalBox, depth: int) -> int:
    source_digits = max(
        (
            canonical_rational_component_digits(endpoint)
            for interval in box.intervals
            for endpoint in (interval.lower, interval.upper)
        ),
        default=1,
    )
    if depth == 0:
        return source_digits
    # A depth-d bisection endpoint is ((2**d-j)l + j*u)/2**d. Before
    # reduction, each component therefore uses at most two source components,
    # a d-bit coefficient, and one carry digit.
    return 2 * source_digits + depth + 2


def _require_deadline(deadline: float, stage: str) -> None:
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"adaptive range enclosure deadline expired {stage}"
        )


def _require_parameter_envelope(problem: _AdaptiveRangeProblem) -> None:
    bounds = (
        ("precision_bits", problem.precision_bits, 32, 4_096),
        (
            "maximum_precision_bits",
            problem.maximum_precision_bits,
            32,
            4_096,
        ),
        ("max_leaves", problem.max_leaves, 1, MAX_ADAPTIVE_RANGE_LEAVES),
        ("max_depth", problem.max_depth, 0, MAX_ADAPTIVE_RANGE_DEPTH),
        (
            "max_evaluations",
            problem.max_evaluations,
            1,
            MAX_ADAPTIVE_RANGE_EVALUATIONS,
        ),
        (
            "wall_seconds",
            problem.wall_seconds,
            1,
            MAX_ADAPTIVE_RANGE_WALL_SECONDS,
        ),
    )
    for field, value, minimum, maximum in bounds:
        if type(value) is not int or not minimum <= value <= maximum:
            raise OperationDomainValidationError(
                location=(field,),
                code="analysis.adaptive_range.parameter_bound",
                message=(f"{field} must be an integer between {minimum} and {maximum}"),
            )
    if problem.maximum_precision_bits < problem.precision_bits:
        raise OperationDomainValidationError(
            location=("maximum_precision_bits", "precision_bits"),
            code="analysis.adaptive_range.precision_schedule",
            message="maximum_precision_bits must be at least precision_bits",
        )


def _require_complete_source_axis(
    problem: _AdaptiveRangeProblem,
    nodes: tuple[IntervalExpressionNode, ...],
) -> None:
    used_variables: set[str] = set()
    for node in nodes:
        if node.op != "var":
            continue
        if node.variable is None:
            raise OperationDomainValidationError(
                location=("expression",),
                code="analysis.adaptive_range.named_variable",
                message="adaptive range variable nodes must be named",
            )
        used_variables.add(node.variable)
    box_variables = set(problem.box.variables)
    missing = used_variables - box_variables
    if missing:
        raise OperationDomainValidationError(
            location=("box", "variables"),
            code="analysis.adaptive_range.missing_variable",
            message=(
                "expression variables are missing from the box: "
                + ", ".join(sorted(missing))
            ),
        )
    unused = box_variables - used_variables
    if unused:
        raise OperationDomainValidationError(
            location=("box", "variables"),
            code="analysis.adaptive_range.unused_variable",
            message=(
                "box variables are unused by the expression: "
                + ", ".join(sorted(unused))
            ),
        )


def _require_source_endpoint_envelope(
    problem: _AdaptiveRangeProblem, *, maximum_leaf_depth: int
) -> None:
    digit_cap = _adaptive_source_endpoint_digit_cap(maximum_leaf_depth)
    if any(
        canonical_rational_component_digits(endpoint) > digit_cap
        for interval in problem.box.intervals
        for endpoint in (interval.lower, interval.upper)
    ):
        raise OperationDomainValidationError(
            location=("box", "intervals"),
            code="analysis.adaptive_range.source_endpoint_digits",
            message=(
                "adaptive source-box endpoints exceed the "
                f"{digit_cap}-digit bound for the reachable split depth "
                f"{maximum_leaf_depth}"
            ),
        )


def _admit_adaptive_range(
    problem: _AdaptiveRangeProblem, *, started_at: float
) -> _AdaptiveRangeAdmission:
    _require_parameter_envelope(problem)
    deadline = started_at + problem.wall_seconds
    bind_request_deadline(deadline)
    _require_deadline(deadline, "before semantic preflight")

    try:
        target_width = problem.target_width.as_fraction()
        if target_width <= 0:
            raise ValueError("adaptive target width must be positive")
        if (
            canonical_rational_component_digits(problem.target_width)
            > MAX_RATIONAL_DIGITS
        ):
            raise ValueError(
                f"adaptive target width exceeds the {MAX_RATIONAL_DIGITS}-digit bound"
            )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("target_width",),
            code="analysis.adaptive_range.target_width",
            message=str(exc),
        ) from exc

    nodes = _bounded_expression_nodes(problem.expression)
    _require_complete_source_axis(problem, nodes)
    plan = _plan_adaptive_range(problem)
    _require_source_endpoint_envelope(
        problem, maximum_leaf_depth=plan.planned_maximum_leaf_depth
    )
    node_evaluations = len(nodes) * plan.planned_evaluations
    if node_evaluations > MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS:
        raise OperationDomainValidationError(
            location=("max_evaluations",),
            code="analysis.adaptive_range.node_evaluations",
            message=(
                f"adaptive range work of {node_evaluations} expression-node "
                f"evaluations exceeds the {MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS}-unit bound"
            ),
        )
    precision_work = len(nodes) * plan.precision_bits_per_expression_node
    if precision_work > MAX_ADAPTIVE_RANGE_PRECISION_WORK:
        raise OperationDomainValidationError(
            location=("maximum_precision_bits", "max_evaluations"),
            code="analysis.adaptive_range.precision_work",
            message=(
                f"adaptive range precision work of {precision_work} node-bit units "
                f"exceeds the {MAX_ADAPTIVE_RANGE_PRECISION_WORK}-unit bound"
            ),
        )
    try:
        preflight = _preflight_box_expression(
            problem.expression, _rational_box_bounds(problem.box)
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("expression",),
            code="analysis.adaptive_range.intermediate_bound",
            message=str(exc),
        ) from exc
    if isinstance(preflight, IntervalExpressionDomainFailure):
        raise OperationDomainValidationError(
            location=("expression", *preflight.node_path),
            code="analysis.adaptive_range.domain_unproven",
            message=(
                "the exact source-box interval extension did not establish the "
                f"real domain for {preflight.operation}"
            ),
        )
    _require_deadline(deadline, "after semantic preflight")
    return _AdaptiveRangeAdmission(
        plan=plan,
        planned_node_evaluations=node_evaluations,
        target_width=target_width,
        deadline=deadline,
    )


def _evaluate_leaf(
    expression: IntervalExpressionNode,
    path: tuple[int, ...],
    box: RationalIntervalBox,
    precision_bits: int,
    *,
    deadline: float,
) -> _EvaluatedAdaptiveRangeLeaf:
    _require_deadline(deadline, "before an Arb leaf evaluation")
    domain_failure: IntervalExpressionDomainFailure | None = None
    try:
        with flint_workprec(precision_bits, deadline=deadline):
            variables = {
                variable: arb_source_interval(interval)
                for variable, interval in zip(box.variables, box.intervals, strict=True)
            }
            result = _evaluate_box_expression(expression, variables)
            if isinstance(result, IntervalExpressionDomainFailure):
                domain_failure = result
                endpoints = None
            elif isinstance(result, _BoxEvaluationFailure) or not result.is_finite():
                raise RuntimeError(
                    "pinned Arb returned no finite adaptive leaf enclosure"
                )
            else:
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
            "pinned Arb rejected an admitted adaptive leaf evaluation"
        ) from exc
    _require_deadline(deadline, "after an Arb leaf evaluation")
    if domain_failure is not None:
        # The exact source-box preflight proved the real domain once. Arb's
        # fixed-precision interval extension can still be unable to prove a
        # domain-sensitive node, so retain that bounded leaf for refinement.
        return _EvaluatedAdaptiveRangeLeaf(
            path=path,
            box=box,
            domain_failure=domain_failure,
        )
    if endpoints is None or any(
        abs(endpoint.exponent) > MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT
        for endpoint in endpoints
    ):
        raise RuntimeError(
            "pinned Arb produced an adaptive endpoint outside the admitted dyadic envelope"
        )
    return _EvaluatedAdaptiveRangeLeaf(
        path=path,
        box=box,
        enclosure=DyadicClosedInterval(lower=endpoints[0], upper=endpoints[1]),
    )


def _public_leaves(
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...],
) -> tuple[AdaptiveRangeFinalLeaf, ...]:
    ordered = tuple(sorted(leaves, key=lambda leaf: leaf.path))
    public: list[AdaptiveRangeFinalLeaf] = []
    for leaf in ordered:
        if leaf.domain_failure is not None:
            public.append(
                AdaptiveRangeDomainUnprovenLeaf(
                    path=leaf.path,
                    box=leaf.box,
                    domain_failure=leaf.domain_failure,
                )
            )
            continue
        assert leaf.enclosure is not None
        public.append(
            AdaptiveRangeLeaf(
                path=leaf.path,
                box=leaf.box,
                enclosure=leaf.enclosure,
            )
        )
    return tuple(public)


def _evaluated_hull(
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...],
) -> DyadicClosedInterval:
    if any(leaf.enclosure is None for leaf in leaves):
        raise AssertionError("a global adaptive hull requires every leaf enclosure")
    lower_leaf = min(
        leaves,
        key=lambda leaf: (
            leaf.enclosure_lower if leaf.enclosure_lower is not None else Fraction()
        ),
    )
    upper_leaf = max(
        leaves,
        key=lambda leaf: (
            leaf.enclosure_upper if leaf.enclosure_upper is not None else Fraction()
        ),
    )
    assert lower_leaf.enclosure is not None
    assert upper_leaf.enclosure is not None
    return DyadicClosedInterval(
        lower=lower_leaf.enclosure.lower,
        upper=upper_leaf.enclosure.upper,
    )


def _result(
    problem: _AdaptiveRangeProblem,
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...],
    *,
    target_width: Fraction,
    evaluations_used: int,
    maximum_precision_bits_used: int,
    reason: AdaptiveRangeBudgetReason | None,
) -> AdaptiveRangeEnclosureResult:
    public_leaves = _public_leaves(leaves)
    has_unproven = any(leaf.domain_failure is not None for leaf in leaves)
    disposition: AdaptiveRangeDisposition
    enclosure: DyadicClosedInterval | None
    if has_unproven:
        assert reason is not None
        enclosure = None
        disposition = AdaptiveRangeDomainUnproven(reason=reason)
    else:
        enclosure = _evaluated_hull(leaves)
        if _enclosure_width(enclosure) <= target_width:
            disposition = AdaptiveRangeTargetMet()
        else:
            assert reason is not None
            disposition = AdaptiveRangeBudgetExhausted(reason=reason)
    return AdaptiveRangeEnclosureResult._from_kernel(
        problem,
        enclosure=enclosure,
        disposition=disposition,
        leaves=public_leaves,
        evaluations_used=evaluations_used,
        maximum_precision_bits_used=maximum_precision_bits_used,
    )


def _finish_result(
    problem: _AdaptiveRangeProblem,
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...],
    *,
    admission: _AdaptiveRangeAdmission,
    evaluations_used: int,
    maximum_precision_bits_used: int,
    reason: AdaptiveRangeBudgetReason | None,
) -> AdaptiveRangeEnclosureResult:
    result = _result(
        problem,
        leaves,
        target_width=admission.target_width,
        evaluations_used=evaluations_used,
        maximum_precision_bits_used=maximum_precision_bits_used,
        reason=reason,
    )
    _require_deadline(admission.deadline, "after result construction")
    return result


def _evaluated_target_met(
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...], target_width: Fraction
) -> bool:
    if any(leaf.enclosure is None for leaf in leaves):
        return False
    return _enclosure_width(_evaluated_hull(leaves)) <= target_width


def _leaf_selection_key(
    leaf: _EvaluatedAdaptiveRangeLeaf,
) -> tuple[int, Fraction, tuple[int, ...]]:
    if leaf.domain_failure is not None:
        return (0, Fraction(), leaf.path)
    assert leaf.enclosure_width is not None
    return (1, -leaf.enclosure_width, leaf.path)


def _run_adaptive_range_enclosure(
    problem: _AdaptiveRangeProblem, *, started_at: float
) -> AdaptiveRangeEnclosureResult:
    admission = _admit_adaptive_range(problem, started_at=started_at)
    schedule = admission.precision_schedule
    evaluations = 0

    root = _evaluate_leaf(
        problem.expression,
        (),
        problem.box,
        schedule[0],
        deadline=admission.deadline,
    )
    evaluations += 1
    precision_used = schedule[0]
    best = root if root.enclosure is not None else None
    leaves: tuple[_EvaluatedAdaptiveRangeLeaf, ...] = (root,)
    if _evaluated_target_met(leaves, admission.target_width):
        return _finish_result(
            problem,
            leaves,
            admission=admission,
            evaluations_used=evaluations,
            maximum_precision_bits_used=precision_used,
            reason=None,
        )

    for precision in schedule[1:]:
        if evaluations >= problem.max_evaluations:
            return _finish_result(
                problem,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=precision_used,
                reason="MAX_EVALUATIONS",
            )
        candidate = _evaluate_leaf(
            problem.expression,
            (),
            problem.box,
            precision,
            deadline=admission.deadline,
        )
        evaluations += 1
        precision_used = precision
        if candidate.enclosure is not None and (
            best is None
            or (
                best.enclosure is not None
                and _enclosure_width(candidate.enclosure)
                < _enclosure_width(best.enclosure)
            )
        ):
            best = candidate
        leaves = (best if best is not None else candidate,)
        if _evaluated_target_met(leaves, admission.target_width):
            return _finish_result(
                problem,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=precision_used,
                reason=None,
            )

    reason: AdaptiveRangeBudgetReason
    while True:
        _require_deadline(admission.deadline, "before partition refinement")
        blocked_unproven = tuple(
            leaf
            for leaf in leaves
            if leaf.domain_failure is not None
            and (len(leaf.path) >= problem.max_depth or leaf.split_coordinate is None)
        )
        if blocked_unproven:
            has_positive_coordinate = any(
                leaf.split_coordinate is not None for leaf in blocked_unproven
            )
            reason = "MAX_DEPTH" if has_positive_coordinate else "MAX_PRECISION"
            break
        candidates = tuple(
            leaf
            for leaf in leaves
            if len(leaf.path) < problem.max_depth and leaf.split_coordinate is not None
        )
        if not candidates:
            any_positive_coordinate = any(
                leaf.split_coordinate is not None for leaf in leaves
            )
            reason = "MAX_DEPTH" if any_positive_coordinate else "MAX_PRECISION"
            break
        if len(leaves) >= problem.max_leaves:
            reason = "MAX_LEAVES"
            break
        if problem.max_evaluations - evaluations < 2:
            reason = "MAX_EVALUATIONS"
            break
        selected = min(
            candidates,
            key=_leaf_selection_key,
        )
        coordinate = selected.split_coordinate
        assert coordinate is not None
        child_boxes = _split_box(selected.box, coordinate)
        children = tuple(
            _evaluate_leaf(
                problem.expression,
                (*selected.path, bit),
                child_box,
                problem.maximum_precision_bits,
                deadline=admission.deadline,
            )
            for bit, child_box in enumerate(child_boxes)
        )
        evaluations += 2
        leaves = tuple(leaf for leaf in leaves if leaf.path != selected.path) + children
        leaves = tuple(sorted(leaves, key=lambda leaf: leaf.path))
        if _evaluated_target_met(leaves, admission.target_width):
            return _finish_result(
                problem,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=problem.maximum_precision_bits,
                reason=None,
            )

    return _finish_result(
        problem,
        leaves,
        admission=admission,
        evaluations_used=evaluations,
        maximum_precision_bits_used=precision_used,
        reason=reason,
    )


def _compute_adaptive_range_enclosure(
    request: AdaptiveRangeEnclosureRequest,
) -> AdaptiveRangeEnclosureResult:
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    return _run_adaptive_range_enclosure(
        _problem_from_request(request),
        started_at=started_at,
    )


def adaptive_range_enclosure(
    expression: IntervalExpressionNode,
    box: RationalIntervalBox,
    target_width: CanonicalRational,
    *,
    precision_bits: int = 128,
    maximum_precision_bits: int = 512,
    max_leaves: int = 32,
    max_depth: int = 8,
    max_evaluations: int = 128,
    wall_seconds: int = 30,
) -> AdaptiveRangeEnclosureResult:
    """Return a sound complete range hull or a typed finite nonconclusion."""

    started_at = monotonic()
    problem = _AdaptiveRangeProblem(
        expression=expression,
        box=box,
        precision_bits=precision_bits,
        target_width=target_width,
        maximum_precision_bits=maximum_precision_bits,
        max_leaves=max_leaves,
        max_depth=max_depth,
        max_evaluations=max_evaluations,
        wall_seconds=wall_seconds,
    )
    return _run_adaptive_range_enclosure(problem, started_at=started_at)


ADAPTIVE_RANGE_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.adaptive_range_enclosure.compute",
        title="Enclose an expression range over an adaptive rational partition",
        description=(
            "Return a sound dyadic hull over every leaf of a deterministic complete "
            "rational-box partition. Precision doubles to the requested maximum; "
            "the narrowest source-box enclosure is retained (earliest precision on "
            "ties). Domain-uncertain leaves are then refined first in canonical path "
            "order; otherwise the widest splittable certified-range leaf is bisected "
            "on its widest source coordinate, with canonical path and axis ties. "
            "TARGET_MET means the hull width is at most target_width. "
            "BUDGET_EXHAUSTED retains the same global soundness after the declared "
            "finite schedule stops. DOMAIN_UNPROVEN carries the complete final cover "
            "but no global enclosure when fixed-precision Arb leaves a local real-"
            "domain obligation uncertain after bounded refinement. The envelope "
            "admits at most "
            f"{MAX_ADAPTIVE_RANGE_LEAVES:,} leaves, depth "
            f"{MAX_ADAPTIVE_RANGE_DEPTH}, {MAX_ADAPTIVE_RANGE_EVALUATIONS:,} "
            "evaluations and 4,096 precision bits, "
            f"{MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS:,} expression-node evaluations, "
            f"{MAX_ADAPTIVE_RANGE_PRECISION_WORK:,} precision-weighted node-bit units."
        ),
        request_type=AdaptiveRangeEnclosureRequest,
        result_type=AdaptiveRangeEnclosureResult,
        run=_compute_adaptive_range_enclosure,
        tags=(
            "analysis",
            "interval",
            "expression",
            "range",
            "adaptive",
            "partition",
            "arb",
            "validated",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="quadratic_unit_interval",
                description=(
                    "Enclose x(1-x) over 0 <= x <= 1 until width 7/16; "
                    "the complete named source axis and finite refinement budgets "
                    "must be supplied."
                ),
                input={
                    "expression": {
                        "op": "mul",
                        "children": [
                            {"op": "var", "variable": "x"},
                            {
                                "op": "sub",
                                "children": [
                                    {
                                        "op": "const",
                                        "value": {"num": "1", "den": "1"},
                                    },
                                    {"op": "var", "variable": "x"},
                                ],
                            },
                        ],
                    },
                    "box": {
                        "variables": ["x"],
                        "intervals": [
                            {
                                "lower": {"num": "0", "den": "1"},
                                "upper": {"num": "1", "den": "1"},
                            }
                        ],
                    },
                    "precision_bits": 128,
                    "target_width": {"num": "7", "den": "16"},
                    "maximum_precision_bits": 128,
                    "max_leaves": 4,
                    "max_depth": 2,
                    "max_evaluations": 7,
                    "wall_seconds": 30,
                },
            ),
        ),
    ),
)


__all__ = [
    "ADAPTIVE_RANGE_ENCLOSURE_OPERATIONS",
    "MAX_ADAPTIVE_RANGE_DEPTH",
    "MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT",
    "MAX_ADAPTIVE_RANGE_EVALUATIONS",
    "MAX_ADAPTIVE_RANGE_LEAVES",
    "MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS",
    "MAX_ADAPTIVE_RANGE_PRECISION_WORK",
    "MAX_ADAPTIVE_RANGE_WALL_SECONDS",
    "AdaptiveRangeBudgetExhausted",
    "AdaptiveRangeDomainUnproven",
    "AdaptiveRangeDomainUnprovenLeaf",
    "AdaptiveRangeEnclosureRequest",
    "AdaptiveRangeEnclosureResult",
    "AdaptiveRangeLeaf",
    "AdaptiveRangeTargetMet",
    "adaptive_range_enclosure",
]
