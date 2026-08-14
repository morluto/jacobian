"""Final projection from authoritative operation state to the v2 wire envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from jacobian.contracts.operations import OperationResult
from jacobian.contracts.results import ContractModel, Execution, ExecutionStatus
from jacobian.operations import Completed, Failed, NonConclusion


@dataclass(frozen=True, slots=True)
class PublishedOperation:
    """One optional public projection and every durable carrier it retained."""

    output: ContractModel | None = None
    artifact_uris: tuple[str, ...] = ()


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
        if (
            not isinstance(self.terminal, Completed)
            and self.publication is not None
            and self.publication.output is not None
        ):
            raise ValueError("non-completed operations cannot publish an output")
        if self.verification_record_uri is not None and (
            not isinstance(self.terminal, Completed)
            or self.publication is None
            or self.publication.output is None
            or self.verification_record_uri not in self.publication.artifact_uris
        ):
            raise ValueError(
                "verification records require completed execution and artifact lineage"
            )


def project_operation_result(projection: OperationProjection) -> OperationResult:
    """Compile one internal operation projection at the public boundary."""

    operation_id = projection.operation_id
    version = projection.version
    terminal = projection.terminal
    publication = projection.publication

    if isinstance(terminal, Completed):
        assert publication is not None
        published_output = publication.output
        assert published_output is not None
        return OperationResult(
            operation_id=operation_id,
            operation_version=version,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=terminal.runtime_ms,
                detail=terminal.detail,
            ),
            output=published_output.model_dump(mode="json"),
            verification_record_uri=projection.verification_record_uri,
            artifact_uris=publication.artifact_uris,
        )

    if isinstance(terminal, (Failed, NonConclusion)):
        status = terminal.status
        diagnostic = terminal.diagnostic
        runtime_ms = terminal.runtime_ms
    else:
        assert_never(terminal)
    if publication is None:
        output = {"error": diagnostic.model_dump(mode="json", exclude_none=True)}
    else:
        output = {}
    return OperationResult(
        operation_id=operation_id,
        operation_version=version,
        execution=Execution(
            status=status,
            runtime_ms=runtime_ms,
            detail=diagnostic.message,
        ),
        output=output,
        diagnostics=(diagnostic,),
        artifact_uris=(publication.artifact_uris if publication is not None else ()),
    )


__all__ = ["OperationProjection", "PublishedOperation", "project_operation_result"]
