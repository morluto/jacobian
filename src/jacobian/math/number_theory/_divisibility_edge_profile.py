"""Divisibility edge profile with quotient and LPF declaration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import CanonicalizationError
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.number_theory._divisibility_edge_profile_kernels import (
    FactorizationIncompleteError,
    construct_divisibility_edge_profile,
)
from jacobian.math.number_theory._divisibility_edge_profile_models import (
    MAX_DIVISIBILITY_EDGE_SET_SIZE,
    DivisibilityEdge,
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
    _validate_divisibility_edge_resources,
    _validate_divisibility_edge_shape,
)

_DIVISIBILITY_EDGE_PROFILE_WALL_SECONDS = 600.0


def _bind_execution_deadline() -> None:
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + _DIVISIBILITY_EDGE_PROFILE_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)


def _require_execution_active(phase: str) -> None:
    request_checkpoint(phase)
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(
            f"divisibility edge profile request deadline expired {phase}"
        )


@contextmanager
def _owner_execution() -> Iterator[None]:
    """Provide one request context for native calls made outside dispatch."""
    if current_request_execution() is None:
        with request_execution(monotonic()):
            yield
    else:
        yield


def compute_divisibility_edge_profile(
    request: DivisibilityEdgeProfileRequest,
) -> DivisibilityEdgeProfileResult:
    """Return the complete directed divisibility edge table with quotient and LPF."""
    with _owner_execution():
        try:
            _bind_execution_deadline()
            _require_execution_active("before admission")
            canonical_elements, edge_plan = _validate_divisibility_edge_resources(
                request.values
            )
            if not canonical_elements:
                return DivisibilityEdgeProfileResult(
                    values=FiniteIntegerSet(elements=()), edges=()
                )
            return _build_divisibility_edge_profile(canonical_elements, edge_plan)
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=("values",), code=exc.type, message=exc.message()
            ) from exc
        except FactorizationIncompleteError as exc:
            failure = exc.failure
            failure_kind = failure.kind if failure is not None else "UNKNOWN"
            raise RuntimeError(
                "divisibility edge factorization worker failed: " + failure_kind
            ) from exc


def _build_divisibility_edge_profile(
    canonical_elements: tuple[int, ...],
    edge_plan: tuple[tuple[int, int, int], ...],
) -> DivisibilityEdgeProfileResult:
    try:
        data = construct_divisibility_edge_profile(canonical_elements, edge_plan)
    except FactorizationIncompleteError as exc:
        failure = exc.failure
        if failure is not None and failure.kind == "WORKER_CANCELLED":
            raise OperationExecutionCancelledError(
                "divisibility edge factorization was cancelled"
            ) from exc
        if failure is not None and failure.kind in {
            "WORKER_TIMEOUT",
            "REQUEST_DEADLINE_EXPIRED",
        }:
            raise OperationExecutionTimeoutError(
                "divisibility edge factorization exceeded its deadline"
            ) from exc
        raise
    edges = tuple(
        DivisibilityEdge(
            source=d.source,
            target=d.target,
            quotient=d.quotient,
            least_prime_factor=d.least_prime_factor,
        )
        for d in data
    )
    result = DivisibilityEdgeProfileResult(
        values=FiniteIntegerSet(elements=canonical_elements), edges=edges
    )
    _require_execution_active("after result construction")
    return result


def divisibility_edge_profile(
    values: FiniteIntegerSet,
) -> DivisibilityEdgeProfileResult:
    """Return a divisibility edge profile from native canonical values."""
    with _owner_execution():
        try:
            _bind_execution_deadline()
            _require_execution_active("before admission")
            if len(values.elements) > MAX_DIVISIBILITY_EDGE_SET_SIZE:
                raise PydanticCustomError(
                    "divisibility_edge.values_size",
                    f"values must contain at most {MAX_DIVISIBILITY_EDGE_SET_SIZE} integers",
                )
            _validate_divisibility_edge_shape(values)
            canonical_elements, edge_plan = _validate_divisibility_edge_resources(
                values
            )
        except (
            CanonicalizationError,
            PydanticCustomError,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(exc, PydanticCustomError):
                code = exc.type
                message = exc.message()
            else:
                code = "divisibility_edge.invalid_native_value"
                message = "values must be canonical integers or IntegerValue instances"
            raise OperationDomainValidationError(
                location=("values",), code=code, message=message
            ) from exc
        try:
            if not canonical_elements:
                return DivisibilityEdgeProfileResult(
                    values=FiniteIntegerSet(elements=()), edges=()
                )
            return _build_divisibility_edge_profile(canonical_elements, edge_plan)
        except FactorizationIncompleteError as exc:
            failure = exc.failure
            failure_kind = failure.kind if failure is not None else "UNKNOWN"
            raise RuntimeError(
                "divisibility edge factorization worker failed: " + failure_kind
            ) from exc


DIVISIBILITY_EDGE_PROFILE_OPERATION = MathTool(
    operation_id="integer.divisibility_edge_profile.compute",
    title="Profile quotient and least-prime-factor data on divisibility edges",
    description="Given an ordered finite source set of positive integers, return the "
    "complete directed proper-divisibility edge table. Each edge a -> b "
    "carries the quotient b/a and its least prime factor.",
    request_type=DivisibilityEdgeProfileRequest,
    result_type=DivisibilityEdgeProfileResult,
    run=compute_divisibility_edge_profile,
    tags=(
        "number-theory",
        "divisibility",
        "primitive-set",
        "least-prime-factor",
        "exact",
    ),
    examples=(
        OperationExample(
            name="divisibility_edges_24612",
            description="For (2,4,6,12), profile all proper-divisibility edges with "
            "quotient and least-prime-factor data; values must be positive "
            "canonical decimal integers.",
            input={"values": {"elements": ["2", "4", "6", "12"]}},
        ),
    ),
)


__all__ = [
    "DIVISIBILITY_EDGE_PROFILE_OPERATION",
    "compute_divisibility_edge_profile",
    "divisibility_edge_profile",
]
