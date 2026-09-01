"""Exact component profiles for bounded plane semialgebraic sets."""

from __future__ import annotations

import os
import re
import time
from fractions import Fraction
from typing import Literal, NoReturn

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    sha256_digest,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicValue
from jacobian.math.polynomials.real_algebra._plane_component_bounds import (
    MAX_PLANE_COMPONENT_PREDICTED_CELLS,
    MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM,
    plane_projection_bound,
    plane_projection_coefficient_bound,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
    MAX_PLANE_COMPONENT_RESULT_BYTES,
    MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_SAMPLE_DEGREE,
    MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL,
    MAX_PLANE_COMPONENT_TOTAL_DEGREE,
    MAX_PLANE_COMPONENT_TOTAL_TERMS,
    PLANE_COMPONENT_WALL_SECONDS,
    IsolatedRealPlanePoint,
    PlaneComponentNoncompletionStatus,
    PlaneComponentProfileComputed,
    PlaneComponentProfileNoncompletion,
    PlaneComponentProfileRequest,
    PlaneComponentProfileResult,
    PlaneSampleDisposition,
    PlaneSemialgebraicComponent,
    PlaneSemialgebraicSet,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_process import (
    QepcadPlaneProcessOutcome,
    QepcadPlaneSampleValidationError,
    run_plane_sample_recognition,
    run_qepcad_plane_components,
)
from jacobian.math.polynomials.values import require_polynomial_budget

_PLANE_COMPONENT_FINALIZATION_SECONDS = 5.0
_PLANE_COMPONENT_OPERATION_VERSION: Literal["1"] = "1"
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"plane_semialgebraic.{reason}", message)


def _require_active(deadline: float, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(
            f"plane-component request cancelled {phase}"
        )
    if deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(
            f"plane-component request deadline expired {phase}"
        )


def _backend_wall_seconds(deadline: float, phase: str) -> float:
    """Leave one bounded interval for result construction after child teardown."""

    available = deadline - time.monotonic()
    if available <= _PLANE_COMPONENT_FINALIZATION_SECONDS:
        raise OperationExecutionTimeoutError(
            f"plane-component request deadline expired before {phase}"
        )
    return available - _PLANE_COMPONENT_FINALIZATION_SECONDS


def _run_admission(request: PlaneComponentProfileRequest) -> None:
    try:
        _admit(request)
    except OperationDomainValidationError:
        raise
    except (PydanticCustomError, ValueError) as exc:
        if isinstance(exc, PydanticCustomError):
            code = exc.type
            message = exc.message()
        else:
            code = "plane_semialgebraic.admission"
            message = str(exc)
        raise OperationDomainValidationError(
            location=(),
            code=code,
            message=message,
        ) from exc


def _admit(request: PlaneComponentProfileRequest) -> None:
    semialgebraic_set = request.semialgebraic_set
    total_terms = sum(
        len(polynomial.polynomial.terms) for polynomial in semialgebraic_set.polynomials
    )
    if total_terms > MAX_PLANE_COMPONENT_TOTAL_TERMS:
        raise _validation_error(
            "total_terms",
            f"plane sign family admits at most {MAX_PLANE_COMPONENT_TOTAL_TERMS} terms",
        )
    for polynomial in semialgebraic_set.polynomials:
        require_polynomial_budget(
            polynomial,
            maximum_terms=MAX_PLANE_COMPONENT_TERMS_PER_POLYNOMIAL,
            maximum_exponent=MAX_PLANE_COMPONENT_TOTAL_DEGREE,
            maximum_coefficient_digits=MAX_PLANE_COMPONENT_COEFFICIENT_DIGITS,
            label="plane sign polynomial",
        )
        if any(
            sum(term.exponents) > MAX_PLANE_COMPONENT_TOTAL_DEGREE
            for term in polynomial.polynomial.terms
        ):
            raise _validation_error(
                "total_degree",
                "plane sign polynomial total degree exceeds the degree-four bound",
            )

    marker_polynomials = tuple(
        polynomial
        for sample in request.samples
        for polynomial in sample.coordinate_polynomials
    )
    for polynomial in marker_polynomials:
        require_polynomial_budget(
            polynomial,
            maximum_terms=MAX_PLANE_COMPONENT_SAMPLE_DEGREE + 1,
            maximum_exponent=MAX_PLANE_COMPONENT_SAMPLE_DEGREE,
            maximum_coefficient_digits=MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS,
            label="plane algebraic sample coordinate",
        )
    for sample in request.samples:
        for interval in sample.isolating_box.intervals:
            require_bounded_rational(
                interval.lower,
                max_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
                label="plane sample isolating endpoint",
            )
            require_bounded_rational(
                interval.upper,
                max_digits=MAX_PLANE_COMPONENT_POINT_ISOLATOR_DIGITS,
                label="plane sample isolating endpoint",
            )

    if not semialgebraic_set.sign_conditions or _whole_plane(semialgebraic_set):
        return

    distinct_markers = tuple(
        {
            encode_strict_json(polynomial.model_dump(mode="json")): polynomial
            for polynomial in marker_polynomials
        }.values()
    )
    projection_degree_sum, predicted_cells = plane_projection_bound(
        (*semialgebraic_set.polynomials, *distinct_markers)
    )
    projected_coefficient_digits = plane_projection_coefficient_bound(
        (*semialgebraic_set.polynomials, *distinct_markers)
    )
    if projection_degree_sum > MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM:
        raise _validation_error(
            "projection_work",
            "plane sign projection degree exceeds the exact CAD work envelope",
        )
    if predicted_cells > MAX_PLANE_COMPONENT_PREDICTED_CELLS:
        raise _validation_error(
            "projection_cells",
            "plane sign CAD cell bound exceeds the exact topology envelope",
        )
    if projected_coefficient_digits > MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS:
        raise _validation_error(
            "projection_height",
            "plane sign projection coefficient height exceeds the exact CAD envelope",
        )


def _whole_plane(semialgebraic_set: PlaneSemialgebraicSet) -> bool:
    return bool(
        len(semialgebraic_set.sign_conditions)
        == 3 ** len(semialgebraic_set.polynomials)
    )


def _rational_point(
    axis: tuple[str, str],
    coordinates: tuple[Fraction, Fraction],
) -> IsolatedRealPlanePoint:
    values: list[RealAlgebraicValue] = []
    intervals: list[ClosedRationalInterval] = []
    for coordinate in coordinates:
        values.append(
            RealAlgebraicValue._from_admitted_polynomial(
                polynomial=(
                    format_canonical_integer(coordinate.denominator),
                    format_canonical_integer(-coordinate.numerator),
                ),
                real_root_index=0,
            )
        )
        endpoint = CanonicalRational.from_fraction(coordinate)
        intervals.append(ClosedRationalInterval(lower=endpoint, upper=endpoint))
    return IsolatedRealPlanePoint(
        axis=axis,
        coordinates=(values[0], values[1]),
        isolating_box=RationalBox(
            domain="QQ",
            variables=axis,
            intervals=(intervals[0], intervals[1]),
        ),
    )


def _noncompletion(
    request: PlaneComponentProfileRequest,
    outcome: QepcadPlaneProcessOutcome,
    *,
    started_at: float | None = None,
    budget_seconds: int,
) -> PlaneComponentProfileResult:
    if outcome.status == "COMPUTED" or outcome.reason is None:
        raise RuntimeError("computed QEPCAD result cannot become a noncompletion")
    status: PlaneComponentNoncompletionStatus = outcome.status
    request_digest = sha256_digest(encode_strict_json(request.model_dump(mode="json")))
    revision = os.environ.get("JACOBIAN_REVISION", "unknown")
    if not _GIT_SHA_PATTERN.fullmatch(revision):
        revision = "unknown"
    timeout_layer: Literal["QEPCAD", "SAMPLE_RECOGNITION"] | None = None
    if status == "TIMEOUT":
        timeout_layer = (
            "SAMPLE_RECOGNITION"
            if outcome.reason is not None and outcome.reason.startswith("SAMPLE_")
            else "QEPCAD"
            if outcome.reason is not None and outcome.reason.startswith("QEPCAD_")
            else None
        )
    elapsed_ms = (
        round(max(0.0, time.monotonic() - started_at) * 1_000)
        if started_at is not None
        else None
    )
    return PlaneComponentProfileResult(
        semialgebraic_set=request.semialgebraic_set,
        samples=request.samples,
        outcome=PlaneComponentProfileNoncompletion(
            status=status,
            reason=outcome.reason,
            request_digest=request_digest,
            budget_seconds=min(budget_seconds, PLANE_COMPONENT_WALL_SECONDS),
            elapsed_ms=elapsed_ms,
            timeout_layer=timeout_layer,
            operation_version=_PLANE_COMPONENT_OPERATION_VERSION,
            repository_revision=revision,
        ),
    )


def _computed_result(
    request: PlaneComponentProfileRequest,
    outcome: PlaneComponentProfileComputed,
    *,
    budget_seconds: int = int(PLANE_COMPONENT_WALL_SECONDS),
) -> PlaneComponentProfileResult:
    """Keep a computed profile only when its exact public value is deliverable."""

    result = PlaneComponentProfileResult(
        semialgebraic_set=request.semialgebraic_set,
        samples=request.samples,
        outcome=outcome,
    )
    try:
        encode_strict_json(
            result.model_dump(mode="json"),
            limits=CanonicalLimits(max_output_bytes=MAX_PLANE_COMPONENT_RESULT_BYTES),
        )
    except CanonicalizationError:
        return _noncompletion(
            request,
            QepcadPlaneProcessOutcome(
                status="RESOURCE_LIMIT",
                reason="RESULT_OUTPUT_LIMIT",
            ),
            budget_seconds=budget_seconds,
        )
    return result


def _raise_sample_domain_error(exc: QepcadPlaneSampleValidationError) -> NoReturn:
    reason = (
        "sample_recognition_bound"
        if exc.reason == "SAMPLE_RECOGNITION_LIMIT"
        else "sample_isolation"
    )
    raise OperationDomainValidationError(
        location=("samples",),
        code=f"plane_semialgebraic.{reason}",
        message=(
            "a supplied plane sample exceeded the exact recognition envelope"
            if exc.reason == "SAMPLE_RECOGNITION_LIMIT"
            else "each supplied plane sample must select one exact real root per coordinate"
        ),
    ) from exc


def compute_plane_component_profile(
    semialgebraic_set: PlaneSemialgebraicSet,
    samples: tuple[IsolatedRealPlanePoint, ...] = (),
) -> PlaneComponentProfileResult:
    """Compute one exact bounded connected-component profile in ``R^2``."""

    request = PlaneComponentProfileRequest(
        semialgebraic_set=semialgebraic_set,
        samples=samples,
    )
    execution = current_request_execution()
    started = time.monotonic() if execution is None else execution.started_at
    owner_deadline = started + PLANE_COMPONENT_WALL_SECONDS
    deadline = (
        owner_deadline
        if execution is None or execution.deadline is None
        else min(owner_deadline, execution.deadline)
    )
    bind_request_deadline(deadline)
    effective_budget_seconds = max(1, round(deadline - started))
    _require_active(deadline, "before semantic admission")
    _run_admission(request)
    _require_active(deadline, "after semantic admission")

    semialgebraic_set = request.semialgebraic_set
    validated_canonical_samples = None
    if request.samples:
        try:
            sample_outcome = run_plane_sample_recognition(
                request,
                wall_seconds=_backend_wall_seconds(
                    deadline, "exact sample recognition"
                ),
            )
        except QepcadPlaneSampleValidationError as exc:
            _raise_sample_domain_error(exc)
        _require_active(deadline, "after exact sample recognition")
        if sample_outcome.status != "COMPUTED":
            return _noncompletion(
                request,
                sample_outcome,
                started_at=started,
                budget_seconds=effective_budget_seconds,
            )
        if sample_outcome.canonical_samples is None:
            return _noncompletion(
                request,
                QepcadPlaneProcessOutcome(
                    status="BACKEND_ERROR",
                    reason="SAMPLE_RECOGNITION_INVALID_OUTPUT",
                ),
                started_at=started,
                budget_seconds=effective_budget_seconds,
            )
        validated_canonical_samples = sample_outcome.canonical_samples

    if not semialgebraic_set.sign_conditions:
        result = _computed_result(
            request,
            PlaneComponentProfileComputed(
                components=(),
                sample_dispositions=tuple(
                    PlaneSampleDisposition(sample_index=index, status="OUTSIDE")
                    for index in range(len(request.samples))
                ),
            ),
            budget_seconds=effective_budget_seconds,
        )
        _require_active(deadline, "after exact result construction")
        return result
    if _whole_plane(semialgebraic_set):
        representative = _rational_point(
            semialgebraic_set.axis, (Fraction(), Fraction())
        )
        result = _computed_result(
            request,
            PlaneComponentProfileComputed(
                components=(
                    PlaneSemialgebraicComponent(
                        component_id=0,
                        representative=representative,
                    ),
                ),
                sample_dispositions=tuple(
                    PlaneSampleDisposition(
                        sample_index=index,
                        status="INSIDE",
                        component_id=0,
                    )
                    for index in range(len(request.samples))
                ),
            ),
            budget_seconds=effective_budget_seconds,
        )
        _require_active(deadline, "after exact result construction")
        return result

    try:
        outcome = run_qepcad_plane_components(
            request,
            wall_seconds=_backend_wall_seconds(
                deadline, "QEPCAD plane-component launch"
            ),
            canonical_samples=validated_canonical_samples,
        )
    except QepcadPlaneSampleValidationError as exc:
        _raise_sample_domain_error(exc)
    _require_active(deadline, "after QEPCAD execution")
    if outcome.status != "COMPUTED":
        return _noncompletion(
            request,
            outcome,
            started_at=started,
            budget_seconds=effective_budget_seconds,
        )
    projection = outcome.projection
    if (
        projection is None
        or len(projection.sample_component_ids) != len(request.samples)
        or any(
            representative.axis != semialgebraic_set.axis
            for representative in projection.representatives
        )
    ):
        return _noncompletion(
            request,
            QepcadPlaneProcessOutcome(
                status="BACKEND_ERROR",
                reason="QEPCAD_INVALID_OUTPUT",
            ),
            started_at=started,
            budget_seconds=effective_budget_seconds,
        )
    result = _computed_result(
        request,
        PlaneComponentProfileComputed(
            components=tuple(
                PlaneSemialgebraicComponent(
                    component_id=component_id,
                    representative=representative,
                )
                for component_id, representative in enumerate(
                    projection.representatives
                )
            ),
            sample_dispositions=tuple(
                PlaneSampleDisposition(
                    sample_index=sample_index,
                    status="OUTSIDE" if component_id is None else "INSIDE",
                    component_id=component_id,
                )
                for sample_index, component_id in enumerate(
                    projection.sample_component_ids
                )
            ),
        ),
    )
    _require_active(deadline, "after exact result construction")
    return result


__all__ = [
    "MAX_PLANE_COMPONENT_PREDICTED_CELLS",
    "MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM",
    "MAX_PLANE_COMPONENT_SAMPLE_COEFFICIENT_DIGITS",
    "MAX_PLANE_COMPONENT_SAMPLE_DEGREE",
    "PLANE_COMPONENT_WALL_SECONDS",
    "compute_plane_component_profile",
]
