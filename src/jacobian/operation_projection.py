"""Final projection from authoritative operation state to the v2 wire envelope."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.contracts.capabilities import CapabilityResult
from jacobian.contracts.results import ContractModel, Execution, ExecutionStatus
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed, NonConclusion


@dataclass(frozen=True, slots=True)
class OperationProjection:
    """Typed execution and publication facts awaiting public dispatch."""

    operation_id: str
    version: str
    terminal: Completed[ContractModel] | NonConclusion | Failed
    publication: PublishedOperation | None = None
    verification_record_uri: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.terminal, Completed) and (
            self.publication is None or self.publication.output is None
        ):
            raise ValueError("completed operations require a public output")
        if self.verification_record_uri is not None and (
            not isinstance(self.terminal, Completed)
            or self.publication is None
            or self.publication.output is None
            or self.verification_record_uri not in self.publication.artifact_uris
        ):
            raise ValueError(
                "verification records require completed execution and artifact lineage"
            )


def project_operation_result(projection: OperationProjection) -> CapabilityResult:
    """Compile one internal operation projection at the public boundary."""

    operation_id = projection.operation_id
    version = projection.version
    terminal = projection.terminal
    publication = projection.publication

    if isinstance(terminal, Completed):
        assert publication is not None
        published_output = publication.output
        assert published_output is not None
        return CapabilityResult(
            capability_id=operation_id,
            capability_version=version,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=terminal.runtime_ms,
                detail=terminal.detail,
            ),
            output=published_output.model_dump(mode="json"),
            verification_record_uri=projection.verification_record_uri,
            artifact_uris=publication.artifact_uris,
        )

    if isinstance(terminal, Failed):
        status = terminal.status
        diagnostic = terminal.diagnostic
        runtime_ms = terminal.runtime_ms
    else:
        status = terminal.status
        diagnostic = terminal.diagnostic
        runtime_ms = terminal.runtime_ms
    if publication is None:
        output = {"error": diagnostic.model_dump(mode="json", exclude_none=True)}
    elif publication.output is None:
        output = {}
    else:
        output = publication.output.model_dump(mode="json")
    return CapabilityResult(
        capability_id=operation_id,
        capability_version=version,
        execution=Execution(
            status=status,
            runtime_ms=runtime_ms,
            detail=diagnostic.message,
        ),
        output=output,
        diagnostics=(diagnostic,),
        artifact_uris=(publication.artifact_uris if publication is not None else ()),
    )


__all__ = ["OperationProjection", "project_operation_result"]
