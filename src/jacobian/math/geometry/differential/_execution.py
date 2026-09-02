"""One request-scoped execution deadline for rational Lie derivatives."""

from __future__ import annotations

from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)

# The maximum dense tensor fixture takes about twelve seconds on the ordinary
# development host.  Ten times that measured duration leaves a generous
# platform margin while keeping every non-cooperative exact backend killable.
LIE_DERIVATIVE_WALL_SECONDS = 120.0


def begin_lie_derivative_deadline() -> float:
    """Bind and return the one deadline shared by all mandatory phases."""

    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    operation_deadline = started_at + LIE_DERIVATIVE_WALL_SECONDS
    deadline = (
        min(operation_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else operation_deadline
    )
    bind_request_deadline(deadline)
    require_lie_derivative_deadline(deadline, "before semantic preflight")
    return deadline


def require_lie_derivative_deadline(deadline: float, stage: str) -> None:
    """Fail without a mathematical conclusion when cancellation or time wins."""

    request_checkpoint(stage)
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"rational Lie derivative deadline expired {stage}"
        )


__all__ = [
    "LIE_DERIVATIVE_WALL_SECONDS",
    "begin_lie_derivative_deadline",
    "require_lie_derivative_deadline",
]
