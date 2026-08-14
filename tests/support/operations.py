"""Operation test builders shared across semantic test lanes."""

from __future__ import annotations

from typing import Any, Protocol

from jacobian.contracts.operations import (
    OperationRequest,
    OperationResult,
)
from jacobian.runtime.services import CoreServices


class OperationRuntime(Protocol):
    """The minimal public invocation surface used by operation tests."""

    @property
    def core(self) -> CoreServices: ...


def invoke_operation(
    runtime: OperationRuntime,
    operation_id: str,
    payload: dict[str, Any],
) -> OperationResult:
    """Invoke one operation through the public request envelope."""

    return runtime.core.operations.invoke(
        OperationRequest(operation_id=operation_id, input=payload)
    )
