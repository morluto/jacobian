"""Transport-only publication for completed ordinary operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.contracts.domain_operations import (
    DurableOperationOutput,
    InlineOperationOutput,
    ReferencedInlineOperationOutput,
)
from jacobian.contracts.results import ContractModel
from jacobian.operation_bindings import (
    DurablePublication,
    InlinePublication,
    InstalledOperation,
)
from jacobian.value_references import ValueReferenceError, ValueReferenceStore


@dataclass(frozen=True, slots=True)
class PublicationContext:
    """Runtime resources required only while projecting a completed value."""

    artifacts: ArtifactService
    values: ValueReferenceStore
    semantics_uri: str
    input_schema_uri: str
    result_schema_uri: str
    backend_version: str


@dataclass(frozen=True, slots=True)
class PublishedOperation:
    """One public projection and every durable carrier it created."""

    output: ContractModel
    artifact_uris: tuple[str, ...] = ()


class PublicationLimitError(Exception):
    """A completed value cannot be exposed within its transport bound."""


def publish_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
    operation: InstalledOperation[RequestT, ResultT],
    request: RequestT,
    result: ResultT,
    context: PublicationContext,
) -> PublishedOperation:
    """Apply only the installed operation's transport publication policy."""

    policy = operation.publication
    if isinstance(policy, InlinePublication):
        projected_payload = {
            "result": result.model_dump(mode="json"),
            "backend_version": context.backend_version,
            "value_refs": {
                port.name: f"value://{'0' * 32}" for port in operation.output_ports
            },
        }
        if len(canonicalize_json(projected_payload)) > policy.maximum_bytes:
            raise PublicationLimitError(
                f"inline result exceeds {policy.maximum_bytes} canonical bytes"
            )
        value_refs: dict[str, str] = {}
        try:
            for port in operation.output_ports:
                value = port.extract_from_result(result)
                value_refs[port.name] = context.values.put(
                    value,
                    operation_id=operation.spec.operation_id,
                    operation_version=operation.spec.version,
                    output_port=port.name,
                )
        except ValueReferenceError as exc:
            raise PublicationLimitError(str(exc)) from exc
        output_contract = (
            ReferencedInlineOperationOutput
            if operation.output_ports
            else InlineOperationOutput
        )
        output_type = cast(Any, output_contract[operation.spec.result_type])
        output_payload = {
            "result": result,
            "backend_version": context.backend_version,
        }
        if value_refs:
            output_payload["value_refs"] = value_refs
        output = cast(
            ContractModel,
            output_type.model_validate(output_payload),
        )
        return PublishedOperation(
            output=output,
        )
    if not isinstance(policy, DurablePublication):
        raise TypeError(f"unsupported publication policy: {type(policy).__name__}")

    input_uri = context.artifacts.put(
        schema_uri=context.input_schema_uri,
        semantics_uri=context.semantics_uri,
        payload=request.model_dump(mode="json"),
        summary=f"{operation.spec.operation_id} materialized input",
    ).artifact_uri
    result_uri = context.artifacts.put(
        schema_uri=context.result_schema_uri,
        semantics_uri=context.semantics_uri,
        payload=result.model_dump(mode="json"),
        parents=(input_uri,),
        summary=f"{operation.spec.operation_id} materialized result",
    ).artifact_uri
    preview_type = policy.preview_type or operation.spec.result_type
    preview = (
        preview_type.model_validate(policy.preview(result))
        if policy.preview is not None
        else None
    )
    durable_output_type = cast(Any, DurableOperationOutput[preview_type])  # type: ignore[valid-type]
    return PublishedOperation(
        output=cast(
            ContractModel,
            durable_output_type.model_validate(
                {
                    "input_uri": input_uri,
                    "result_uri": result_uri,
                    "preview": preview,
                    "preview_complete": policy.preview_complete,
                    "backend_version": context.backend_version,
                }
            ),
        ),
        artifact_uris=(input_uri, result_uri),
    )


__all__ = [
    "PublicationContext",
    "PublicationLimitError",
    "PublishedOperation",
    "publish_operation",
]
