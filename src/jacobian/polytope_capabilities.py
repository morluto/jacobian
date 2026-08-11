"""Direct installation of the retained finite-polytope operation."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.polytope import PolytopeSeparateRequest, PolytopeSeparateResult
from jacobian.contracts.results import Coverage, ExecutionStatus
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
        envelope = value.result
        scope = CapabilityScope(
            description=(
                f"scope reported with {envelope.assurance.coverage.value} coverage"
            ),
            parameters={
                "point_uri": value.point_uri,
                "generator_set_uri": value.generator_set_uri,
                "projection": parsed.projection,
            },
            artifact_uri=envelope.assurance.scope_uri,
        )
        complete = (
            envelope.execution.status is ExecutionStatus.COMPLETED
            and envelope.assurance.coverage is Coverage.EXHAUSTIVE
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=envelope.execution,
            output=value.model_dump(mode="json"),
            scope=scope,
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if complete
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=(
                    f"underlying result reports {envelope.assurance.coverage.value} "
                    "coverage over the declared scope"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic local finite-polytope computation",
            ),
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
