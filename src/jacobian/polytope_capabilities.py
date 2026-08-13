"""Direct installation of the retained finite-polytope operation."""

from __future__ import annotations

from jacobian.capability_adapters import parse_capability_input
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.contracts.polytope import PolytopeSeparateRequest, PolytopeSeparateResult
from jacobian.contracts.results import ExecutionStatus
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed, Failed
from jacobian.polytope import PolytopeService
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema

__all__ = ["PolytopeSeparationAdapter"]


class PolytopeSeparationAdapter:
    """Expose exact separation without the deleted generic service adapter."""

    def __init__(self, service: PolytopeService) -> None:
        self._service = service
        self._descriptor = CapabilityDescriptor(
            capability_id="polytope.separate",
            version="1",
            title="Separate a rational point from a convex hull",
            description=(
                "Compute exact membership evidence or a separator; replay is separate."
            ),
            provider="jacobian.z3",
            provider_runtime=known_provider_runtime(
                "jacobian.z3",
                features=("polytope", "exact"),
            ),
            input_schema=model_schema(PolytopeSeparateRequest),
            output_schema=model_schema(PolytopeSeparateResult),
            tags=("polytope", "exact"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> PolytopeSeparateRequest:
        return parse_capability_input(PolytopeSeparateRequest, request.input)

    def invoke(self, parsed: PolytopeSeparateRequest) -> OperationProjection:
        value = self._service.separate(parsed)
        terminal = (
            Completed(
                value=value,
                runtime_ms=value.execution.runtime_ms,
                detail=value.execution.detail,
            )
            if value.execution.status is ExecutionStatus.COMPLETED
            else Failed(
                status=value.execution.status,
                runtime_ms=value.execution.runtime_ms,
                diagnostic=CapabilityDiagnostic(
                    code="POLYTOPE_SEPARATION_NOT_COMPLETED",
                    stage="solver_execution",
                    message=(
                        value.execution.detail
                        or "The exact polytope operation did not complete."
                    ),
                ),
            )
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=terminal,
            publication=PublishedOperation(
                output=value if isinstance(terminal, Completed) else None,
                artifact_uris=_artifact_references(value),
            ),
        )


def _artifact_references(value: PolytopeSeparateResult) -> tuple[str, ...]:
    references = [value.point_uri, value.generator_set_uri]
    for reference in (
        value.effective_point_uri,
        value.effective_generator_set_uri,
        value.claim_uri,
        value.witness_uri,
        value.certificate_uri,
    ):
        if reference is not None:
            references.append(reference)
    return tuple(references)
