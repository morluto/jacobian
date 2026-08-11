"""Atomic capability adapters over explicitly installed application services.

The adapters intentionally expose individual materialization, evaluation,
search, and replay operations.  They do not compose those operations into a
verification workflow: search output remains computed evidence until an
explicit checker-backed capability accepts it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from jacobian.atomic_domain_capabilities import build_domain_adapters
from jacobian.atomic_experiment_capabilities import build_experiment_adapters
from jacobian.contracts.artifacts import ArtifactPutResult
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityProviderAvailability,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.claims import ClaimValidationResult
from jacobian.contracts.conjectures import ParameterRegionEvidence
from jacobian.contracts.evaluation import EvaluationBatchResult, EvaluationProfile
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.results import (
    Coverage,
    Execution,
    ExecutionStatus,
    ResultEnvelope,
    Verification,
)
from jacobian.contracts.shrinking import ShrinkResult
from jacobian.contracts.witness_search import WitnessFindResult
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository

if TYPE_CHECKING:
    from jacobian.installation.context import InstallationContext
    from jacobian.runtime.services import ApplicationServices


_ARTIFACT_URI_PATTERN = r"^artifact://sha256/[0-9a-f]{64}$"
_CHECKER_URI_PATTERN = r"^checker://sha256/[0-9a-f]{64}$"
_EXPERIMENT_URI_PATTERN = r"^experiment://[0-9a-f]{32}$"
_ARTIFACT_URI = {"type": "string", "pattern": _ARTIFACT_URI_PATTERN}
_CHECKER_URI = {"type": "string", "pattern": _CHECKER_URI_PATTERN}
_EXPERIMENT_URI = {"type": "string", "pattern": _EXPERIMENT_URI_PATTERN}


class AtomicServiceAdapter:
    """Project one service operation into the capability protocol.

    This adapter is reserved for existing stateful services that already
    return rich result envelopes. Domain-owned mathematical producers use
    ``DomainBundle`` and ``OperationInstaller`` instead.

    A service result may carry a nested :class:`ResultEnvelope`; only that
    envelope (or an explicitly promoted parameter region) can elevate the
    capability assurance.  This keeps evaluator, enumerator, and solver
    output from self-certifying merely because it found useful evidence.
    """

    def __init__(
        self,
        *,
        capability_id: str,
        title: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        invoke: Callable[[dict[str, Any]], Any],
        store: ArtifactRepository,
        artifact_references: Callable[[Any], tuple[str, ...]] | None = None,
        unverified_assurance_level: CapabilityAssuranceLevel = (
            CapabilityAssuranceLevel.COMPUTED
        ),
        unverified_basis: str = "deterministic local service result",
        read_only: bool = False,
        discovery_visible: bool = True,
        tags: tuple[str, ...] = (),
        provider: str = "jacobian.runtime",
    ) -> None:
        self._descriptor = CapabilityDescriptor(
            capability_id=capability_id,
            version="1",
            title=title,
            description=description,
            provider=provider,
            provider_runtime=known_provider_runtime(provider, features=tags),
            input_schema=input_schema,
            output_schema=output_schema,
            read_only=read_only,
            discovery_visible=discovery_visible,
            tags=tags,
        )
        self._invoke = invoke
        self._store = store
        self._artifact_references = artifact_references
        self._unverified_assurance_level = unverified_assurance_level
        self._unverified_basis = unverified_basis

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        value = self._invoke(request.input)
        output = _dump(value)
        execution = _execution(value)
        record_uri = _verified_record_uri(value)
        artifact_uris = (
            self._artifact_references(value)
            if self._artifact_references is not None
            else ()
        )
        verification_artifacts: tuple[str, ...] | None = None
        if record_uri is not None:
            verification_artifacts = _verification_bindings(
                record_uri,
                artifact_uris,
                self._store,
            )
            if verification_artifacts is not None:
                artifact_uris = verification_artifacts
        verified = (
            execution.status is ExecutionStatus.COMPLETED
            and record_uri is not None
            and verification_artifacts is not None
            and _is_verified(value)
        )
        scope, completeness = _scope_and_completeness(
            value,
            request_input=request.input,
            verified=verified,
            assurance_level=(
                CapabilityAssuranceLevel.VERIFIED
                if verified
                else self._unverified_assurance_level
            ),
            verification_record_uri=record_uri if verified else None,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=execution,
            output=output,
            scope=scope,
            completeness=completeness,
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else self._unverified_assurance_level
                ),
                basis=(
                    "accepted by an authorized independent checker"
                    if verified
                    else self._unverified_basis
                ),
                verification_record_uri=record_uri if verified else None,
            ),
            artifact_uris=artifact_uris,
        )


def install_atomic_capabilities(
    context: InstallationContext,
    application: ApplicationServices,
) -> tuple[AtomicServiceAdapter, ...]:
    """Build the bundled atomic adapters without adding MCP-specific behavior."""

    def _adapter(**kwargs: Any) -> AtomicServiceAdapter:
        return AtomicServiceAdapter(store=context.store, **kwargs)

    adapters = (
        _adapter(
            capability_id="artifact.put",
            title="Store a schema-validated artifact",
            description="Materialize one immutable artifact with explicit lineage.",
            input_schema=_schema(
                {
                    "schema_uri": _ARTIFACT_URI,
                    "semantics_uri": _ARTIFACT_URI,
                    "payload": {},
                    "parents": {
                        "type": "array",
                        "items": _ARTIFACT_URI,
                        "maxItems": 1024,
                    },
                    "summary": {"type": "string", "maxLength": 512},
                },
                required=("schema_uri", "semantics_uri", "payload"),
            ),
            output_schema=model_schema(ArtifactPutResult),
            invoke=lambda p: context.artifacts.put(
                schema_uri=p["schema_uri"],
                semantics_uri=p["semantics_uri"],
                payload=p["payload"],
                parents=tuple(p.get("parents", ())),
                summary=p.get("summary", ""),
            ),
            artifact_references=lambda v: (v.artifact_uri,),
            discovery_visible=False,
            tags=("artifact", "storage"),
        ),
        _adapter(
            capability_id="claim.validate",
            title="Validate a claim against one plugin",
            description="Check claim schema, semantics, and declared plugin capabilities.",
            input_schema=_schema(
                {"claim_uri": _ARTIFACT_URI, "plugin_id": _ARTIFACT_URI},
                required=("claim_uri", "plugin_id"),
            ),
            output_schema=model_schema(ClaimValidationResult),
            invoke=lambda p: application.claims.validate(**p),
            artifact_references=lambda v: (v.claim_uri, v.plugin_id),
            read_only=True,
            tags=("claim", "validation"),
        ),
        _adapter(
            capability_id="evaluate.batch",
            title="Evaluate candidates",
            description="Run a plugin evaluator over a bounded batch without verification.",
            input_schema=_schema(
                {
                    "claim_uri": _ARTIFACT_URI,
                    "candidate_uris": {
                        "type": "array",
                        "items": _ARTIFACT_URI,
                        "minItems": 1,
                        "maxItems": 256,
                    },
                    "plugin_id": _ARTIFACT_URI,
                    "profile": {"enum": ["FAST", "EXACT_CANDIDATE"]},
                    "seed": {"type": "integer"},
                    "wall_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 86400,
                    },
                },
                required=(
                    "claim_uri",
                    "candidate_uris",
                    "plugin_id",
                    "profile",
                    "seed",
                    "wall_seconds",
                ),
            ),
            output_schema=model_schema(EvaluationBatchResult),
            invoke=lambda p: application.evaluation.evaluate_batch(
                **{
                    **p,
                    "candidate_uris": tuple(p["candidate_uris"]),
                    "profile": EvaluationProfile(p["profile"]),
                }
            ),
            artifact_references=lambda v: (
                v.claim_uri,
                v.plugin_id,
                *(item.candidate_uri for item in v.items),
            ),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="untrusted plugin evaluation is not independently verified",
            tags=("evaluation",),
        ),
        _adapter(
            capability_id="witness.find",
            title="Find one witness",
            description="Search for a witness or a bounded no-witness certificate proposal.",
            input_schema=_schema(
                {
                    "claim_uri": _ARTIFACT_URI,
                    "candidate_uri": _ARTIFACT_URI,
                    "plugin_id": _ARTIFACT_URI,
                    "witness_role": {
                        "enum": [
                            "DEFEATS_CANDIDATE",
                            "RESCUES_CANDIDATE",
                            "SUPPORTS_CLAIM",
                            "REFUTES_CLAIM",
                        ]
                    },
                    "wall_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 86400,
                    },
                },
                required=(
                    "claim_uri",
                    "candidate_uri",
                    "plugin_id",
                    "witness_role",
                    "wall_seconds",
                ),
            ),
            output_schema=model_schema(WitnessFindResult),
            invoke=lambda p: application.witnesses.find(
                **{**p, "witness_role": WitnessRole(p["witness_role"])}
            ),
            artifact_references=lambda v: _witness_find_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="witness search output is evidence pending explicit replay",
            tags=("witness", "search"),
        ),
        _adapter(
            capability_id="witness.verify",
            title="Verify one witness",
            description="Replay one witness with an explicitly selected authorized checker.",
            input_schema=_verification_schema(
                {
                    "claim_uri": _ARTIFACT_URI,
                    "candidate_uri": _ARTIFACT_URI,
                    "witness_uri": _ARTIFACT_URI,
                    "checker_id": _CHECKER_URI,
                },
                required=("claim_uri", "candidate_uri", "witness_uri", "checker_id"),
            ),
            output_schema=model_schema(ResultEnvelope),
            invoke=lambda p: application.verification.verify_witness(**p),
            artifact_references=lambda v: _envelope_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="the checker did not accept the supplied witness",
            tags=("witness", "verification"),
        ),
        _adapter(
            capability_id="certificate.verify",
            title="Verify one certificate",
            description="Replay one certificate with a compatible authorized checker.",
            input_schema=_verification_schema(
                {"certificate_uri": _ARTIFACT_URI, "checker_id": _CHECKER_URI},
                required=("certificate_uri",),
            ),
            output_schema=model_schema(ResultEnvelope),
            invoke=lambda p: application.verification.verify_certificate(**p),
            artifact_references=lambda v: _envelope_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="the checker did not accept the supplied certificate",
            tags=("certificate", "verification"),
        ),
        _adapter(
            capability_id="shrink.run",
            title="Shrink a candidate or witness",
            description="Apply bounded reductions and replay each accepted preservation claim.",
            input_schema=_schema(
                {
                    "target_kind": {"enum": ["candidate", "witness"]},
                    "target_uri": _ARTIFACT_URI,
                    "claim_uri": _ARTIFACT_URI,
                    "plugin_id": _ARTIFACT_URI,
                    "preservation_checker_id": _CHECKER_URI,
                    "reducers": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        "minItems": 1,
                        "maxItems": 128,
                    },
                    "objectives": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        "maxItems": 128,
                    },
                    "evaluation_budget": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000000,
                    },
                },
                required=(
                    "target_kind",
                    "target_uri",
                    "claim_uri",
                    "plugin_id",
                    "preservation_checker_id",
                    "reducers",
                    "objectives",
                    "evaluation_budget",
                ),
            ),
            output_schema=model_schema(ShrinkResult),
            invoke=lambda p: application.shrinking.run(
                **{
                    **p,
                    "reducers": tuple(p["reducers"]),
                    "objectives": tuple(p["objectives"]),
                }
            ),
            artifact_references=lambda v: _shrink_references(v),
            unverified_assurance_level=CapabilityAssuranceLevel.HEURISTIC,
            unverified_basis="plugin-proposed reductions are not a verified minimality claim",
            tags=("shrink",),
        ),
        *build_experiment_adapters(
            application,
            adapter=_adapter,
            schema=_schema,
            artifact_uri=_ARTIFACT_URI,
            experiment_uri=_EXPERIMENT_URI,
            enumeration_budget_schema=_enumeration_budget_schema,
        ),
        *build_domain_adapters(
            application,
            adapter=_adapter,
            schema=_schema,
            artifact_uri=_ARTIFACT_URI,
        ),
    )
    return tuple(
        adapter
        for adapter in adapters
        if (runtime := adapter.descriptor.provider_runtime) is None
        or runtime.availability is CapabilityProviderAvailability.AVAILABLE
    )


def _schema(properties: dict[str, Any], *, required: Iterable[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _verification_schema(
    properties: dict[str, Any], *, required: Iterable[str]
) -> dict[str, Any]:
    combined = dict(properties)
    combined["timeout_seconds"] = {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 86400,
    }
    return _schema(combined, required=required)


def _enumeration_budget_schema() -> dict[str, Any]:
    return _schema(
        {
            "candidates_max": {"type": "integer", "minimum": 1, "maximum": 10000000},
            "wall_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 4096},
        },
        required=("candidates_max", "wall_seconds"),
    )


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif isinstance(value, dict):
        dumped = value
    else:
        raise TypeError("atomic capability service returned an unsupported value")
    if not isinstance(dumped, dict):
        raise TypeError("atomic capability service returned a non-object value")
    return dumped


def _scope_and_completeness(
    value: Any,
    *,
    request_input: dict[str, Any],
    verified: bool,
    assurance_level: CapabilityAssuranceLevel,
    verification_record_uri: str | None,
) -> tuple[CapabilityScope | None, CapabilityCompleteness]:
    envelope = _result_envelope(value)
    if envelope is None or envelope.assurance.coverage is Coverage.NOT_APPLICABLE:
        return (
            None,
            CapabilityCompleteness(
                basis="the operation makes no completeness claim",
            ),
        )

    scope_parameters = {
        key: item
        for key, item in request_input.items()
        if key.endswith("_uri")
        or key
        in {
            "bounds",
            "candidate_uris",
            "profile",
            "projection",
            "seed",
        }
    }
    scope = (
        CapabilityScope(
            description=(
                f"scope reported with {envelope.assurance.coverage.value} coverage"
            ),
            parameters=scope_parameters,
            artifact_uri=envelope.assurance.scope_uri,
        )
        if scope_parameters or envelope.assurance.scope_uri is not None
        else None
    )
    complete = (
        envelope.execution.status is ExecutionStatus.COMPLETED
        and envelope.assurance.coverage is Coverage.EXHAUSTIVE
        and scope is not None
    )
    checker_bound_scope = (
        envelope.assurance.scope_uri is not None and complete and verified
    )
    completeness_level = (
        CapabilityAssuranceLevel.VERIFIED
        if checker_bound_scope
        else (
            CapabilityAssuranceLevel.COMPUTED
            if assurance_level is CapabilityAssuranceLevel.VERIFIED
            else assurance_level
        )
    )
    return (
        scope,
        CapabilityCompleteness(
            status=(
                CapabilityCompletenessStatus.COMPLETE
                if complete
                else CapabilityCompletenessStatus.PARTIAL
            ),
            basis=(
                f"underlying result reports {envelope.assurance.coverage.value} "
                "coverage over the declared scope"
            ),
            assurance_level=completeness_level,
            verification_record_uri=(
                verification_record_uri
                if completeness_level is CapabilityAssuranceLevel.VERIFIED
                else None
            ),
        ),
    )


def _result_envelope(value: Any) -> ResultEnvelope | None:
    if isinstance(value, ResultEnvelope):
        return value
    nested = getattr(value, "result", None)
    return nested if isinstance(nested, ResultEnvelope) else None


def _execution(value: Any) -> Execution:
    execution = getattr(value, "execution", None)
    if isinstance(execution, Execution):
        return execution
    nested = getattr(value, "result", None)
    if isinstance(nested, ResultEnvelope):
        return nested.execution
    return Execution(status=ExecutionStatus.COMPLETED)


def _has_verified_parameter_region_evidence(value: Any) -> bool:
    evidence = getattr(value, "evidence", None)
    return isinstance(evidence, str) and evidence in {
        ParameterRegionEvidence.VERIFIED_SUFFICIENT,
        ParameterRegionEvidence.VERIFIED_NECESSARY,
    }


def _verified_record_uri(value: Any) -> str | None:
    envelope = _result_envelope(value)
    if envelope is not None:
        return envelope.verification_record_uri
    if _has_verified_parameter_region_evidence(value):
        return getattr(value, "verification_record_uri", None)
    return None


def _is_verified(value: Any) -> bool:
    envelope = _result_envelope(value)
    if isinstance(envelope, ResultEnvelope):
        return envelope.assurance.verification is Verification.VERIFIED
    return _has_verified_parameter_region_evidence(value)


def _witness_find_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = [value.claim_uri, value.candidate_uri, value.plugin_id]
    if value.witness_uri is not None:
        refs.append(value.witness_uri)
    if value.certificate_uri is not None:
        refs.append(value.certificate_uri)
    return tuple(refs)


def _envelope_references(value: Any) -> tuple[str, ...]:
    envelope = _result_envelope(value)
    if envelope is None:
        return ()
    refs: list[str] = list(envelope.evidence_uris)
    if envelope.verification_record_uri is not None:
        refs.append(envelope.verification_record_uri)
    if envelope.assurance.scope_uri is not None:
        refs.append(envelope.assurance.scope_uri)
    return tuple(refs)


def _shrink_references(value: Any) -> tuple[str, ...]:
    refs: list[str] = [value.initial_target_uri, value.final_target_uri]
    for step in value.steps:
        refs.append(step.from_uri)
        if step.proposed_uri is not None:
            refs.append(step.proposed_uri)
        if step.verification_record_uri is not None:
            refs.append(step.verification_record_uri)
    return tuple(refs)


def _verification_bindings(
    record_uri: str,
    artifact_uris: tuple[str, ...],
    store: ArtifactRepository,
) -> tuple[str, ...] | None:
    try:
        record = store.get(record_uri)
    except StorageError:
        return None
    return tuple(sorted({*artifact_uris, record_uri, *record.manifest.parents}))
