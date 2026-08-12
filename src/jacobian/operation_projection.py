"""Temporary projection from authoritative operation state to the v2 wire envelope."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityResult
from jacobian.contracts.results import ContractModel, Execution, ExecutionStatus
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed, NonConclusion


def project_operation_result(
    *,
    operation_id: str,
    version: str,
    terminal: Completed[ContractModel] | NonConclusion | Failed,
    publication: PublishedOperation | None = None,
) -> CapabilityResult:
    """Project one terminal state without making the envelope authoritative."""

    if isinstance(terminal, Completed):
        if publication is None:
            raise ValueError("completed operation requires publication")
        return CapabilityResult(
            capability_id=operation_id,
            capability_version=version,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=terminal.runtime_ms,
            ),
            output=publication.output.model_dump(mode="json"),
            artifact_uris=publication.artifact_uris,
        )

    if isinstance(terminal, Failed):
        status = terminal.status
        diagnostic = terminal.diagnostic
    else:
        status = ExecutionStatus.ERROR
        diagnostic = terminal.diagnostic
    return CapabilityResult(
        capability_id=operation_id,
        capability_version=version,
        execution=Execution(status=status, detail=diagnostic.message),
        output={"error": diagnostic.model_dump(mode="json", exclude_none=True)},
        diagnostics=(diagnostic,),
    )


__all__ = ["project_operation_result"]
