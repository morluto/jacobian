"""Adaptive complete range enclosures over exact rational box partitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.analysis._arb import arb_source_interval, dyadic_endpoints
from jacobian.math.analysis._box_enclosure import (
    _BoxEvaluationFailure,
    _evaluate_box_expression,
    _preflight_box_expression,
)
from jacobian.math.analysis._models import (
    MAX_BOX_PREFLIGHT_TEMPORARY_BITS,
    MAX_DYADIC_MANTISSA_DIGITS,
    MAX_RATIONAL_DIGITS,
    DyadicClosedInterval,
    ExactDyadic,
    IntervalExpressionDomainFailure,
    IntervalExpressionNode,
    RationalIntervalBox,
    _bound_raw_rational,
    _bounded_expression_nodes,
    _IntervalExpressionBoxRequest,
    _rational_box_bounds,
    _validation_error,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval

MAX_ADAPTIVE_RANGE_LEAVES = 1_024
MAX_ADAPTIVE_RANGE_DEPTH = 32
MAX_ADAPTIVE_RANGE_EVALUATIONS = 4_096
MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS = 131_072
MAX_ADAPTIVE_RANGE_PRECISION_WORK = 268_435_456
MAX_ADAPTIVE_RANGE_WALL_SECONDS = 120
MAX_ADAPTIVE_RANGE_RESULT_BYTES = CanonicalLimits().max_output_bytes

# Admission bounds every exact source-box value by an 8,192-bit rational and
# every Arb work precision by 4,096 bits. An outward endpoint can therefore
# require at most their sum in its binary exponent. The existing temporary
# Fraction ceiling is a slightly larger, named bound that also keeps public
# result validation from expanding an enormous authored dyadic exponent.
MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT = MAX_BOX_PREFLIGHT_TEMPORARY_BITS

type AdaptiveRangeBudgetReason = Literal[
    "MAX_LEAVES", "MAX_DEPTH", "MAX_EVALUATIONS", "MAX_PRECISION"
]


class AdaptiveRangeTargetMet(StrictModel):
    """The complete enclosure is no wider than the requested target."""

    status: Literal["TARGET_MET"] = "TARGET_MET"


class AdaptiveRangeBudgetExhausted(StrictModel):
    """The complete enclosure is sound but the finite schedule stopped first."""

    status: Literal["BUDGET_EXHAUSTED"] = "BUDGET_EXHAUSTED"
    reason: AdaptiveRangeBudgetReason


type AdaptiveRangeDisposition = Annotated[
    AdaptiveRangeTargetMet | AdaptiveRangeBudgetExhausted,
    Field(discriminator="status"),
]


class AdaptiveRangeEnclosureRequest(_IntervalExpressionBoxRequest):
    """Bound one complete elementary-expression range by adaptive bisection."""

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
        return value

    @model_validator(mode="after")
    def require_precision_schedule(self) -> Self:
        if self.maximum_precision_bits < self.precision_bits:
            raise _validation_error(
                "maximum_precision_bits must be at least precision_bits"
            )
        return self


class AdaptiveRangeLeaf(StrictModel):
    """One canonically addressed leaf and its sound expression enclosure."""

    path: tuple[StrictInt, ...] = Field(
        max_length=MAX_ADAPTIVE_RANGE_DEPTH,
        description=(
            "Binary child choices from the source box. At every step 0 selects "
            "the lower and 1 the upper exact-midpoint child of the widest positive "
            "coordinate, with source-axis ties."
        ),
    )
    box: RationalIntervalBox
    enclosure: DyadicClosedInterval

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


def _precision_schedule(initial: int, maximum: int) -> tuple[int, ...]:
    schedule = [initial]
    while schedule[-1] < maximum:
        schedule.append(min(2 * schedule[-1], maximum))
    return tuple(schedule)


def _dyadic_fraction(value: ExactDyadic) -> Fraction:
    if abs(value.exponent) > MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT:
        raise _validation_error(
            "adaptive dyadic exponent exceeds the admitted source-and-precision bound"
        )
    return value.as_fraction()


def _enclosure_width(enclosure: DyadicClosedInterval) -> Fraction:
    return _dyadic_fraction(enclosure.upper) - _dyadic_fraction(enclosure.lower)


def _interval_hull(leaves: tuple[AdaptiveRangeLeaf, ...]) -> DyadicClosedInterval:
    lower = min(
        (leaf.enclosure.lower for leaf in leaves),
        key=_dyadic_fraction,
    )
    upper = max(
        (leaf.enclosure.upper for leaf in leaves),
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


def _box_at_path(
    source: RationalIntervalBox, path: tuple[int, ...]
) -> RationalIntervalBox:
    box = source
    for bit in path:
        coordinate = _widest_coordinate(box)
        if coordinate is None:
            raise _validation_error(
                "a positive-depth leaf cannot descend from a degenerate box"
            )
        children = _split_box(box, coordinate)
        box = children[bit]
    return box


def _paths_are_complete(paths: tuple[tuple[int, ...], ...]) -> bool:
    for index, path in enumerate(paths):
        if any(other[: len(path)] == path for other in paths[index + 1 :]):
            return False
    depth = max(map(len, paths), default=0)
    return sum(1 << (depth - len(path)) for path in paths) == 1 << depth


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
        if leaf.box != _box_at_path(result.box, leaf.path):
            raise _validation_error(
                "adaptive leaf box does not match its source-bound midpoint path"
            )
        _enclosure_width(leaf.enclosure)


def _bind_evaluation_schedule(
    result: AdaptiveRangeEnclosureResult, schedule: tuple[int, ...]
) -> int:
    root_evaluations = result.evaluations_used - 2 * (len(result.leaves) - 1)
    if not 1 <= root_evaluations <= len(schedule):
        raise _validation_error(
            "adaptive evaluation count does not reconstruct its finite schedule"
        )
    if result.maximum_precision_bits_used != schedule[root_evaluations - 1]:
        raise _validation_error(
            "adaptive used precision does not match its doubling schedule"
        )
    if len(result.leaves) > 1 and root_evaluations != len(schedule):
        raise _validation_error(
            "adaptive bisection may begin only after the precision schedule"
        )
    if result.evaluations_used > result.max_evaluations:
        raise _validation_error(
            "adaptive result exceeds the requested evaluation budget"
        )
    return root_evaluations


def _bind_budget_reason(
    result: AdaptiveRangeEnclosureResult,
    schedule: tuple[int, ...],
    root_evaluations: int,
) -> None:
    if not isinstance(result.disposition, AdaptiveRangeBudgetExhausted):
        return
    reason = result.disposition.reason
    if reason != "MAX_EVALUATIONS" and root_evaluations != len(schedule):
        raise _validation_error(
            "only MAX_EVALUATIONS may stop an incomplete precision schedule"
        )
    if reason == "MAX_LEAVES" and len(result.leaves) != result.max_leaves:
        raise _validation_error(
            "MAX_LEAVES requires the requested leaf budget to be full"
        )
    if reason == "MAX_EVALUATIONS":
        remaining = result.max_evaluations - result.evaluations_used
        required = 1 if root_evaluations < len(schedule) else 2
        if remaining >= required:
            raise _validation_error(
                "MAX_EVALUATIONS must lack the next complete evaluation action"
            )
    if reason not in ("MAX_DEPTH", "MAX_PRECISION"):
        return
    splittable = any(
        len(leaf.path) < result.max_depth and _widest_coordinate(leaf.box) is not None
        for leaf in result.leaves
    )
    if splittable:
        raise _validation_error(f"{reason} cannot stop while a leaf remains splittable")
    has_positive_coordinate = any(
        _widest_coordinate(leaf.box) is not None for leaf in result.leaves
    )
    if (reason == "MAX_DEPTH") != has_positive_coordinate:
        raise _validation_error(
            "MAX_DEPTH and MAX_PRECISION must identify the deterministic no-split cause"
        )
    if reason == "MAX_PRECISION" and (
        result.maximum_precision_bits_used != result.maximum_precision_bits
    ):
        raise _validation_error(
            "MAX_PRECISION requires completing the precision schedule"
        )


class AdaptiveRangeEnclosureResult(AdaptiveRangeEnclosureRequest):
    """A complete source-bound range enclosure and exact finite leaf cover."""

    enclosure: DyadicClosedInterval
    disposition: AdaptiveRangeDisposition
    leaves: tuple[AdaptiveRangeLeaf, ...] = Field(
        min_length=1,
        max_length=MAX_ADAPTIVE_RANGE_LEAVES,
        description=(
            "Canonical path-ordered complete partition. Leaf interiors are "
            "disjoint; adjacent closed leaves share their bisection face."
        ),
    )
    evaluations_used: StrictInt = Field(ge=1, le=MAX_ADAPTIVE_RANGE_EVALUATIONS)
    maximum_precision_bits_used: StrictInt = Field(ge=32, le=4096)

    @field_validator("leaves", mode="before")
    @classmethod
    def preserve_bounded_leaves(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and len(value) > MAX_ADAPTIVE_RANGE_LEAVES:
            raise _validation_error("adaptive result exceeds its leaf bound")
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bind_partition_and_hull_to_source(self) -> Self:
        _bind_leaf_partition(self)
        hull = _interval_hull(self.leaves)
        if self.enclosure != hull:
            raise _validation_error(
                "adaptive global enclosure must equal the hull of every leaf enclosure"
            )
        target = self.target_width.as_fraction()
        if target <= 0:
            raise _validation_error("adaptive target width must be positive")
        target_met = _enclosure_width(hull) <= target
        if target_met != isinstance(self.disposition, AdaptiveRangeTargetMet):
            raise _validation_error(
                "adaptive disposition must agree with the requested target width"
            )

        schedule = _precision_schedule(self.precision_bits, self.maximum_precision_bits)
        root_evaluations = _bind_evaluation_schedule(self, schedule)
        _bind_budget_reason(self, schedule, root_evaluations)
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: AdaptiveRangeEnclosureRequest,
        *,
        enclosure: DyadicClosedInterval,
        disposition: AdaptiveRangeDisposition,
        leaves: tuple[AdaptiveRangeLeaf, ...],
        evaluations_used: int,
        maximum_precision_bits_used: int,
    ) -> AdaptiveRangeEnclosureResult:
        """Construct a result after the kernel established every range claim."""

        return cls.model_construct(
            expression=request.expression,
            box=request.box,
            precision_bits=request.precision_bits,
            target_width=request.target_width,
            maximum_precision_bits=request.maximum_precision_bits,
            max_leaves=request.max_leaves,
            max_depth=request.max_depth,
            max_evaluations=request.max_evaluations,
            wall_seconds=request.wall_seconds,
            enclosure=enclosure,
            disposition=disposition,
            leaves=leaves,
            evaluations_used=evaluations_used,
            maximum_precision_bits_used=maximum_precision_bits_used,
        )


@dataclass(frozen=True, slots=True)
class _AdaptiveRangeAdmission:
    precision_schedule: tuple[int, ...]
    planned_evaluations: int
    target_width: Fraction
    deadline: float


def _midpoint_component_digits(box: RationalIntervalBox, depth: int) -> int:
    source_digits = max(
        (
            canonical_rational_component_digits(endpoint)
            for interval in box.intervals
            for endpoint in (interval.lower, interval.upper)
        ),
        default=1,
    )
    # A depth-d bisection endpoint is ((2**d-j)l + j*u)/2**d. Before
    # reduction, each component therefore uses at most two source components,
    # a d-bit coefficient, and one carry digit.
    return 2 * source_digits + depth + 2


def _estimated_result_bytes(request: AdaptiveRangeEnclosureRequest) -> int:
    source_bytes = len(canonicalize_json(request.model_dump(mode="json")))
    endpoint_digits = _midpoint_component_digits(request.box, request.max_depth)
    axis_bytes = sum(
        len(variable.encode("utf-8")) + 3 for variable in request.box.variables
    )
    box_bytes = (
        128 + axis_bytes + len(request.box.variables) * (4 * endpoint_digits + 128)
    )
    dyadic_interval_bytes = 2 * (MAX_DYADIC_MANTISSA_DIGITS + 64) + 64
    path_bytes = 2 * request.max_depth + 32
    leaf_bytes = box_bytes + dyadic_interval_bytes + path_bytes + 128
    return source_bytes + request.max_leaves * leaf_bytes + 4_096


def _require_deadline(deadline: float, stage: str) -> None:
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"adaptive range enclosure deadline expired {stage}"
        )


def _admit_adaptive_range(
    request: AdaptiveRangeEnclosureRequest, *, started_at: float
) -> _AdaptiveRangeAdmission:
    deadline = started_at + request.wall_seconds
    bind_request_deadline(deadline)
    _require_deadline(deadline, "before semantic preflight")

    try:
        target_width = request.target_width.as_fraction()
        if target_width <= 0:
            raise ValueError("adaptive target width must be positive")
        if (
            canonical_rational_component_digits(request.target_width)
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

    nodes = _bounded_expression_nodes(request.expression)
    schedule = _precision_schedule(
        request.precision_bits, request.maximum_precision_bits
    )
    planned_evaluations = min(
        request.max_evaluations,
        len(schedule) + 2 * (request.max_leaves - 1),
    )
    node_evaluations = len(nodes) * planned_evaluations
    if node_evaluations > MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS:
        raise OperationDomainValidationError(
            location=("max_evaluations",),
            code="analysis.adaptive_range.node_evaluations",
            message=(
                f"adaptive range work of {node_evaluations} expression-node "
                f"evaluations exceeds the {MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS}-unit bound"
            ),
        )
    precision_work = node_evaluations * request.maximum_precision_bits
    if precision_work > MAX_ADAPTIVE_RANGE_PRECISION_WORK:
        raise OperationDomainValidationError(
            location=("maximum_precision_bits", "max_evaluations"),
            code="analysis.adaptive_range.precision_work",
            message=(
                f"adaptive range precision work of {precision_work} node-bit units "
                f"exceeds the {MAX_ADAPTIVE_RANGE_PRECISION_WORK}-unit bound"
            ),
        )
    estimated_result_bytes = _estimated_result_bytes(request)
    if estimated_result_bytes > MAX_ADAPTIVE_RANGE_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("max_leaves", "max_depth", "box"),
            code="analysis.adaptive_range.result_bytes",
            message=(
                f"adaptive range result estimate of {estimated_result_bytes} bytes "
                f"exceeds the {MAX_ADAPTIVE_RANGE_RESULT_BYTES}-byte canonical output bound"
            ),
        )

    try:
        preflight = _preflight_box_expression(
            request.expression, _rational_box_bounds(request.box)
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
        precision_schedule=schedule,
        planned_evaluations=planned_evaluations,
        target_width=target_width,
        deadline=deadline,
    )


def _evaluate_leaf(
    expression: IntervalExpressionNode,
    box: RationalIntervalBox,
    precision_bits: int,
    *,
    deadline: float,
) -> DyadicClosedInterval:
    _require_deadline(deadline, "before an Arb leaf evaluation")
    try:
        from flint import ctx

        with ctx.workprec(precision_bits):
            variables = {
                variable: arb_source_interval(interval)
                for variable, interval in zip(box.variables, box.intervals, strict=True)
            }
            result = _evaluate_box_expression(expression, variables)
            if isinstance(result, IntervalExpressionDomainFailure):
                raise RuntimeError(
                    "pinned Arb disagreed with the admitted real source domain"
                )
            if isinstance(result, _BoxEvaluationFailure) or not result.is_finite():
                raise RuntimeError(
                    "pinned Arb returned no finite adaptive leaf enclosure"
                )
            lower_mantissa, lower_exponent = result.lower().man_exp()
            upper_mantissa, upper_exponent = result.upper().man_exp()
            endpoints = dyadic_endpoints(
                lower_mantissa,
                lower_exponent,
                upper_mantissa,
                upper_exponent,
            )
    except OperationExecutionTimeoutError:
        raise
    except RuntimeError:
        raise
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(
            "pinned Arb rejected an admitted adaptive leaf evaluation"
        ) from exc
    _require_deadline(deadline, "after an Arb leaf evaluation")
    if endpoints is None or any(
        abs(endpoint.exponent) > MAX_ADAPTIVE_RANGE_DYADIC_EXPONENT
        for endpoint in endpoints
    ):
        raise RuntimeError(
            "pinned Arb produced an adaptive endpoint outside the admitted dyadic envelope"
        )
    return DyadicClosedInterval(lower=endpoints[0], upper=endpoints[1])


def _result(
    request: AdaptiveRangeEnclosureRequest,
    leaves: tuple[AdaptiveRangeLeaf, ...],
    *,
    target_width: Fraction,
    evaluations_used: int,
    maximum_precision_bits_used: int,
    reason: AdaptiveRangeBudgetReason | None,
) -> AdaptiveRangeEnclosureResult:
    leaves = tuple(sorted(leaves, key=lambda leaf: leaf.path))
    hull = _interval_hull(leaves)
    target_met = _enclosure_width(hull) <= target_width
    disposition: AdaptiveRangeDisposition
    if target_met:
        disposition = AdaptiveRangeTargetMet()
    else:
        assert reason is not None
        disposition = AdaptiveRangeBudgetExhausted(reason=reason)
    return AdaptiveRangeEnclosureResult._from_kernel(
        request,
        enclosure=hull,
        disposition=disposition,
        leaves=leaves,
        evaluations_used=evaluations_used,
        maximum_precision_bits_used=maximum_precision_bits_used,
    )


def _finish_result(
    request: AdaptiveRangeEnclosureRequest,
    leaves: tuple[AdaptiveRangeLeaf, ...],
    *,
    admission: _AdaptiveRangeAdmission,
    evaluations_used: int,
    maximum_precision_bits_used: int,
    reason: AdaptiveRangeBudgetReason | None,
) -> AdaptiveRangeEnclosureResult:
    result = _result(
        request,
        leaves,
        target_width=admission.target_width,
        evaluations_used=evaluations_used,
        maximum_precision_bits_used=maximum_precision_bits_used,
        reason=reason,
    )
    _require_deadline(admission.deadline, "after result construction")
    return result


def _compute_adaptive_range_enclosure(
    request: AdaptiveRangeEnclosureRequest, *, native_started_at: float | None = None
) -> AdaptiveRangeEnclosureResult:
    execution = current_request_execution()
    started_at = (
        execution.started_at
        if execution is not None
        else native_started_at
        if native_started_at is not None
        else monotonic()
    )
    admission = _admit_adaptive_range(request, started_at=started_at)
    schedule = admission.precision_schedule
    evaluations = 0

    root_path: tuple[int, ...] = ()
    best = _evaluate_leaf(
        request.expression,
        request.box,
        schedule[0],
        deadline=admission.deadline,
    )
    evaluations += 1
    precision_used = schedule[0]
    leaves: tuple[AdaptiveRangeLeaf, ...] = (
        AdaptiveRangeLeaf(path=root_path, box=request.box, enclosure=best),
    )
    if _enclosure_width(best) <= admission.target_width:
        return _finish_result(
            request,
            leaves,
            admission=admission,
            evaluations_used=evaluations,
            maximum_precision_bits_used=precision_used,
            reason=None,
        )

    for precision in schedule[1:]:
        if evaluations >= request.max_evaluations:
            return _finish_result(
                request,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=precision_used,
                reason="MAX_EVALUATIONS",
            )
        candidate = _evaluate_leaf(
            request.expression,
            request.box,
            precision,
            deadline=admission.deadline,
        )
        evaluations += 1
        precision_used = precision
        if _enclosure_width(candidate) < _enclosure_width(best):
            best = candidate
            leaves = (
                AdaptiveRangeLeaf(path=root_path, box=request.box, enclosure=best),
            )
        if _enclosure_width(best) <= admission.target_width:
            return _finish_result(
                request,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=precision_used,
                reason=None,
            )

    while True:
        _require_deadline(admission.deadline, "before partition refinement")
        if len(leaves) >= request.max_leaves:
            reason: AdaptiveRangeBudgetReason = "MAX_LEAVES"
            break
        if request.max_evaluations - evaluations < 2:
            reason = "MAX_EVALUATIONS"
            break

        candidates = tuple(
            leaf
            for leaf in leaves
            if len(leaf.path) < request.max_depth
            and _widest_coordinate(leaf.box) is not None
        )
        if not candidates:
            any_positive_coordinate = any(
                _widest_coordinate(leaf.box) is not None for leaf in leaves
            )
            reason = "MAX_DEPTH" if any_positive_coordinate else "MAX_PRECISION"
            break
        selected = min(
            candidates,
            key=lambda leaf: (-_enclosure_width(leaf.enclosure), leaf.path),
        )
        coordinate = _widest_coordinate(selected.box)
        assert coordinate is not None
        child_boxes = _split_box(selected.box, coordinate)
        child_enclosures = tuple(
            _evaluate_leaf(
                request.expression,
                child_box,
                request.maximum_precision_bits,
                deadline=admission.deadline,
            )
            for child_box in child_boxes
        )
        evaluations += 2
        children = tuple(
            AdaptiveRangeLeaf(
                path=(*selected.path, bit),
                box=child_box,
                enclosure=enclosure,
            )
            for bit, (child_box, enclosure) in enumerate(
                zip(child_boxes, child_enclosures, strict=True)
            )
        )
        leaves = tuple(leaf for leaf in leaves if leaf.path != selected.path) + children
        leaves = tuple(sorted(leaves, key=lambda leaf: leaf.path))
        if _enclosure_width(_interval_hull(leaves)) <= admission.target_width:
            return _finish_result(
                request,
                leaves,
                admission=admission,
                evaluations_used=evaluations,
                maximum_precision_bits_used=request.maximum_precision_bits,
                reason=None,
            )

    return _finish_result(
        request,
        leaves,
        admission=admission,
        evaluations_used=evaluations,
        maximum_precision_bits_used=precision_used,
        reason=reason,
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
    """Return a sound complete range hull over a deterministic finite cover."""

    started_at = monotonic()
    request = AdaptiveRangeEnclosureRequest(
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
    return _compute_adaptive_range_enclosure(request, native_started_at=started_at)


ADAPTIVE_RANGE_ENCLOSURE_OPERATIONS = (
    MathTool(
        operation_id="interval.expression.adaptive_range_enclosure.compute",
        title="Enclose an expression range over an adaptive rational partition",
        description=(
            "Return a sound dyadic hull over every leaf of a deterministic complete "
            "rational-box partition. Precision doubles to the requested maximum; "
            "the narrowest source-box enclosure is retained (earliest precision on "
            "ties), then the widest currently splittable certified-range leaf is "
            "bisected on its widest source coordinate, with canonical path and axis "
            "ties. TARGET_MET means the hull width is at most target_width. "
            "BUDGET_EXHAUSTED retains the same global soundness after the declared "
            "finite schedule stops. The envelope admits at most "
            f"{MAX_ADAPTIVE_RANGE_LEAVES:,} leaves, depth "
            f"{MAX_ADAPTIVE_RANGE_DEPTH}, {MAX_ADAPTIVE_RANGE_EVALUATIONS:,} "
            "evaluations and 4,096 precision bits, "
            f"{MAX_ADAPTIVE_RANGE_NODE_EVALUATIONS:,} expression-node evaluations, "
            f"{MAX_ADAPTIVE_RANGE_PRECISION_WORK:,} precision-weighted node-bit "
            "units, and one complete result within the 10 MiB canonical output bound."
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
            example(
                "quadratic_unit_interval",
                (
                    "Enclose x(1-x) over 0 <= x <= 1 until width 7/16; "
                    "the complete named source axis and finite refinement budgets "
                    "must be supplied."
                ),
                {
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
    "MAX_ADAPTIVE_RANGE_RESULT_BYTES",
    "MAX_ADAPTIVE_RANGE_WALL_SECONDS",
    "AdaptiveRangeBudgetExhausted",
    "AdaptiveRangeEnclosureRequest",
    "AdaptiveRangeEnclosureResult",
    "AdaptiveRangeLeaf",
    "AdaptiveRangeTargetMet",
    "adaptive_range_enclosure",
]
