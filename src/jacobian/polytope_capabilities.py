"""Direct installation of the retained finite-polytope operation."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.polytope import PolytopeSeparateRequest, PolytopeSeparateResult
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

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        parsed = PolytopeSeparateRequest.model_validate(request.input)
        value = self._service.separate(parsed)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=value.execution,
            output=value.model_dump(mode="json"),
            artifact_uris=_artifact_references(value),
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
