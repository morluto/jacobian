"""Atomic transformation and specialized-domain capability registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jacobian.atomic_capability_builders import AdapterFactory, SchemaBuilder
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
)
from jacobian.contracts.conjectures import ParameterRegion
from jacobian.contracts.polytope import PolytopeSeparateRequest, PolytopeSeparateResult
from jacobian.contracts.results import ResultEnvelope
from jacobian.contracts.transformations import (
    TransformationApplyResult,
    TransformationRelation,
)
from jacobian.schema_registry import model_schema

if TYPE_CHECKING:
    from jacobian.atomic_capabilities import AtomicServiceAdapter
    from jacobian.runtime.services import ApplicationServices


def build_domain_adapters(
    application: ApplicationServices,
    *,
    adapter: AdapterFactory,
    schema: SchemaBuilder,
    artifact_uri: dict[str, Any],
) -> tuple[AtomicServiceAdapter, ...]:
    """Build transformation, polytope, and parameter-region adapters."""

    return (
        adapter(
            capability_id="transform.apply",
            title="Apply one representation transformation",
            description=(
                "Materialize a plugin-proposed transformation and its verification "
                "obligation."
            ),
            input_schema=schema(
                {
                    "source_uri": artifact_uri,
                    "plugin_id": artifact_uri,
                    "target_schema_uri": artifact_uri,
                    "target_semantics_uri": artifact_uri,
                    "requested_relation": {
                        "enum": [
                            "EQUIVALENT",
                            "OVER_APPROXIMATION",
                            "UNDER_APPROXIMATION",
                            "HEURISTIC",
                        ]
                    },
                    "wall_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 86400,
                    },
                },
                required=(
                    "source_uri",
                    "plugin_id",
                    "target_schema_uri",
                    "target_semantics_uri",
                    "requested_relation",
                    "wall_seconds",
                ),
            ),
            output_schema=model_schema(TransformationApplyResult),
            invoke=lambda p: application.transformations.apply(
                **{
                    **p,
                    "requested_relation": TransformationRelation(
                        p["requested_relation"]
                    ),
                }
            ),
            artifact_references=lambda v: _transformation_apply_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis=(
                "plugin transformation output remains an open verification obligation"
            ),
            tags=("transform",),
        ),
        adapter(
            capability_id="transform.verify",
            title="Verify one transformation",
            description=(
                "Replay one transformation relation with its compatible authorized "
                "checker."
            ),
            input_schema=schema(
                {"transformation_uri": artifact_uri},
                required=("transformation_uri",),
            ),
            output_schema=model_schema(ResultEnvelope),
            invoke=lambda p: application.verification.verify_transformation(**p),
            artifact_references=lambda v: _envelope_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis=("the checker did not accept the transformation relation"),
            tags=("transform", "verification"),
        ),
        adapter(
            capability_id="polytope.separate",
            title="Separate a rational point from a convex hull",
            description=(
                "Compute exact membership evidence or a separator; replay is separate."
            ),
            input_schema=schema(
                {
                    "point_uri": artifact_uri,
                    "generator_set_uri": artifact_uri,
                    "projection": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                        "maxItems": 256,
                        "uniqueItems": True,
                    },
                    "wall_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 86400,
                    },
                },
                required=("point_uri", "generator_set_uri"),
            ),
            output_schema=model_schema(PolytopeSeparateResult),
            invoke=lambda p: application.polytope.separate(
                PolytopeSeparateRequest(**p)
            ),
            artifact_references=lambda v: _polytope_separate_references(v),
            tags=("polytope", "exact"),
            provider="jacobian.z3",
        ),
        adapter(
            capability_id="parameter.region.promote",
            title="Promote one verified parameter region",
            description=(
                "Replay a record bound to an immutable region before marking it "
                "verified."
            ),
            input_schema=schema(
                {
                    "subject_uri": artifact_uri,
                    "verification_record_uri": artifact_uri,
                },
                required=("subject_uri", "verification_record_uri"),
            ),
            output_schema=model_schema(ParameterRegion),
            invoke=lambda p: application.conjectures.promote_parameter_region(**p),
            artifact_references=lambda v: _parameter_region_references(v),
            read_only=True,
            tags=("parameter", "verification"),
        ),
    )


def _transformation_apply_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = [value.source_uri]
    if value.target_uri is not None:
        refs.append(value.target_uri)
    if value.claim_uri is not None:
        refs.append(value.claim_uri)
    if value.transformation_uri is not None:
        refs.append(value.transformation_uri)
    return tuple(refs)


def _envelope_references(value: Any) -> tuple[str, ...]:
    envelope = getattr(value, "result", None)
    if not isinstance(envelope, ResultEnvelope):
        return ()
    refs: list[str] = list(envelope.evidence_uris)
    if envelope.verification_record_uri is not None:
        refs.append(envelope.verification_record_uri)
    if envelope.assurance.scope_uri is not None:
        refs.append(envelope.assurance.scope_uri)
    return tuple(refs)


def _polytope_separate_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = [value.point_uri, value.generator_set_uri]
    if value.effective_point_uri is not None:
        refs.append(value.effective_point_uri)
    if value.effective_generator_set_uri is not None:
        refs.append(value.effective_generator_set_uri)
    if value.claim_uri is not None:
        refs.append(value.claim_uri)
    if value.witness_uri is not None:
        refs.append(value.witness_uri)
    if value.certificate_uri is not None:
        refs.append(value.certificate_uri)
    return tuple(refs)


def _parameter_region_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = list(value.sample_uris)
    if value.subject_uri is not None:
        refs.append(value.subject_uri)
    if value.verification_record_uri is not None:
        refs.append(value.verification_record_uri)
    return tuple(refs)
