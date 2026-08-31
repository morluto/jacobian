"""Native invariant integral-form lattice operation."""

from __future__ import annotations

from functools import partial
from time import monotonic

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.lattices.invariant_forms._kernel import (
    _admit_invariant_bilinear_form_lattice,
    invariant_bilinear_form_lattice_kernel,
)
from jacobian.math.lattices.invariant_forms._models import (
    FormKind,
    InvariantBilinearFormLattice,
    MatrixAction,
)

_INVARIANT_FORM_WALL_SECONDS = 600.0


def _execution_deadline() -> float:
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    owner_deadline = started_at + _INVARIANT_FORM_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _require_deadline(deadline: float, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError(f"invariant-form deadline expired {phase}")


def compute_invariant_bilinear_form_lattice(
    action: MatrixAction,
    kind: FormKind,
) -> InvariantBilinearFormLattice:
    """Compute all integral forms ``Q`` with ``A^T Q A = Q`` exactly."""

    if kind not in ("BILINEAR", "SYMMETRIC", "ALTERNATING"):
        raise OperationDomainValidationError(
            location=("kind",),
            code="lattice.invariant_form.invalid_kind",
            message="kind must be BILINEAR, SYMMETRIC, or ALTERNATING",
        )
    try:
        deadline = _execution_deadline()
        execution_checkpoint = partial(_require_deadline, deadline)
        execution_checkpoint("before invariant-form semantic admission")
        admission = _admit_invariant_bilinear_form_lattice(
            action,
            kind,
        )
        execution_checkpoint("after invariant-form semantic admission")
        return invariant_bilinear_form_lattice_kernel(
            action,
            kind,
            admission=admission,
            execution_checkpoint=execution_checkpoint,
            deadline=deadline,
        )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("action",),
            code=exc.type,
            message=exc.message(),
        ) from exc


__all__ = ["compute_invariant_bilinear_form_lattice"]
