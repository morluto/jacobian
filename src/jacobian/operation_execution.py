"""Authoritative terminal-state execution for ordinary mathematical operations."""

from __future__ import annotations

import time

from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ContractModel
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operation_declarations import (
    PreflightStatus as DeclarationPreflightStatus,
)
from jacobian.operations import (
    Completed,
    Failed,
    NonConclusion,
    OperationAbortError,
    OperationRefusalError,
    PreflightStatus,
)

type OperationTerminal[ResultT: ContractModel] = (
    Completed[ResultT] | NonConclusion | Failed
)


def execute_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    spec: OperationDeclaration[RequestT, ResultT],
    request: RequestT,
) -> OperationTerminal[ResultT]:
    """Run preflight, execution, result parsing, and postcondition exactly once."""

    if spec.preflight is not None:
        preflight = spec.preflight(request)
        if preflight.status not in {
            PreflightStatus.SUPPORTED,
            DeclarationPreflightStatus.SUPPORTED,
        }:
            return NonConclusion(
                OperationDiagnostic(
                    code=preflight.status.value,
                    stage="operation_preflight",
                    message=preflight.reason or "Operation preflight rejected.",
                )
            )

    started = time.monotonic()
    try:
        outcome = spec.execute(request)
    except OperationRefusalError as exc:
        return NonConclusion(exc.diagnostic)
    except OperationAbortError as exc:
        return Failed(status=exc.status, diagnostic=exc.diagnostic)
    result = spec.result_type.model_validate(outcome)
    if spec.postcondition is not None:
        spec.postcondition(request, result)
    return Completed(
        value=result,
        runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
    )


__all__ = ["OperationTerminal", "execute_operation"]
