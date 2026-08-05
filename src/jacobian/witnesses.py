"""Adversarial witness search that never self-certifies."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from jacobian.claims import ClaimValidationService
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.plugins import CapabilityName
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.contracts.witness_search import (
    PluginWitnessResponse,
    WitnessFindResult,
    WitnessSearchStatus,
)
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry, PluginRegistryError
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

_LOGGER = logging.getLogger(__name__)


def _conclusion_proving_absence(role: WitnessRole) -> Conclusion:
    """Return the conclusion required to prove no witness of ``role`` exists."""

    if role in {WitnessRole.DEFEATS_CANDIDATE, WitnessRole.REFUTES_CLAIM}:
        return Conclusion.TRUE
    return Conclusion.FALSE


class WitnessSearchService:
    """Ask an untrusted plugin for adversarial evidence without self-certifying."""

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        verification: VerificationService,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.verification = verification
        self.witness_schema_uri = schemas.register(
            name="jacobian.witness-envelope",
            version="1",
            schema=model_schema(WitnessEnvelope),
        )

    def find(
        self,
        *,
        claim_uri: str,
        candidate_uri: str,
        plugin_id: str,
        witness_role: WitnessRole | str,
        wall_seconds: float,
    ) -> WitnessFindResult:
        """Find a witness or independently verify a no-witness certificate."""

        role = WitnessRole(witness_role)
        validation = self.claims.validate(
            claim_uri=claim_uri,
            plugin_id=plugin_id,
        )
        if not validation.valid:
            return self._rejected(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                plugin_id=plugin_id,
                detail="; ".join(validation.input.errors),
            )
        if wall_seconds < 1:
            return self._rejected(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                plugin_id=plugin_id,
                detail=(
                    "The witness-search time budget must be at least 1 second. "
                    "Set wall_seconds to 1 or more, then retry."
                ),
            )
        try:
            claim = self.store.get(claim_uri)
            candidate = self.store.get(candidate_uri)
            manifest = self.plugins.get(plugin_id)
            if candidate.manifest.schema_uri != manifest.candidate_schema_uri:
                raise ValueError(
                    "The candidate uses the wrong schema for this plugin. Use a "
                    "candidate returned by the same reference domain, then retry."
                )
            if candidate.manifest.semantics_uri != manifest.semantics_uri:
                raise ValueError(
                    "The candidate and plugin use different semantics. Choose the "
                    "plugin from the candidate's reference domain, then retry."
                )
            self.schemas.validate(
                manifest.candidate_schema_uri,
                candidate.payload,
            )
            oracle = self.plugins.resolve(
                plugin_id,
                CapabilityName.WITNESS_ORACLE,
            )
            semantics = self.store.get(manifest.semantics_uri)
            execution = self.executor.run(
                entrypoint=oracle.descriptor.entrypoint,
                implementation_digest=oracle.implementation_digest,
                request={
                    "request_version": "1",
                    "witness_role": role.value,
                    "claim": claim.payload,
                    "candidate": candidate.payload,
                    "bindings": {
                        "claim_digest": claim.manifest.object_digest,
                        "candidate_digest": candidate.manifest.object_digest,
                        "semantics_digest": semantics.manifest.object_digest,
                    },
                },
                timeout_seconds=wall_seconds,
            )
            if execution.status != ExecutionStatus.COMPLETED:
                return WitnessFindResult(
                    status=WitnessSearchStatus.UNKNOWN,
                    result=ResultEnvelope(
                        execution=Execution(
                            status=execution.status,
                            runtime_ms=execution.runtime_ms,
                            detail=execution.detail,
                        ),
                        input=InputValidation(status=InputStatus.ACCEPTED),
                        conclusion=Conclusion.UNKNOWN,
                        assurance=Assurance(
                            arithmetic=Arithmetic.SYMBOLIC,
                            method=Method.BOUNDED_SEARCH,
                            coverage=Coverage.BOUNDED,
                            verification=Verification.UNVERIFIED,
                        ),
                        claim_digest=claim.manifest.object_digest,
                        semantics_digest=semantics.manifest.object_digest,
                        candidate_digest=candidate.manifest.object_digest,
                    ),
                    claim_uri=claim_uri,
                    candidate_uri=candidate_uri,
                    plugin_id=plugin_id,
                    detail=execution.detail or "",
                )

            response = PluginWitnessResponse.model_validate(execution.output)
            witness_uri = None
            if response.status == WitnessSearchStatus.NONE_CERTIFIED:
                if response.certificate_uri is None:
                    raise ValueError("NONE_CERTIFIED response requires certificate_uri")
                verified = self.verification.verify_certificate(
                    certificate_uri=response.certificate_uri
                )
                if (
                    verified.assurance.verification == Verification.VERIFIED
                    and verified.conclusion == _conclusion_proving_absence(role)
                    and verified.claim_digest == claim.manifest.object_digest
                    and verified.semantics_digest == semantics.manifest.object_digest
                    and verified.candidate_digest == candidate.manifest.object_digest
                    and response.certificate_uri in verified.evidence_uris
                ):
                    return WitnessFindResult(
                        status=WitnessSearchStatus.NONE_CERTIFIED,
                        result=verified,
                        claim_uri=claim_uri,
                        candidate_uri=candidate_uri,
                        plugin_id=plugin_id,
                        certificate_uri=response.certificate_uri,
                        detail=response.detail,
                    )
                raise ValueError(
                    "The no-witness certificate was not verified for this claim and "
                    "candidate. Verify a certificate bound to these exact artifacts."
                )
            if response.status == WitnessSearchStatus.FOUND:
                if response.role != role:
                    raise ValueError(
                        "The plugin returned a different witness role than requested. "
                        "Call math.find and retry with a supported role."
                    )
                if (
                    response.witness is None
                    or response.witness_format is None
                    or response.format_version is None
                ):
                    raise ValueError(
                        "The plugin reported FOUND without complete witness data. "
                        "Retry once; if it repeats, inspect the local plugin log."
                    )
                witness = WitnessEnvelope(
                    witness_format=response.witness_format,
                    format_version=response.format_version,
                    role=response.role,
                    bindings=EvidenceBindings(
                        claim_digest=claim.manifest.object_digest,
                        semantics_digest=semantics.manifest.object_digest,
                        candidate_digest=candidate.manifest.object_digest,
                    ),
                    payload=response.witness,
                )
                stored_witness = self.store.put(
                    schema_uri=self.witness_schema_uri,
                    semantics_uri=manifest.semantics_uri,
                    payload=witness.model_dump(mode="json"),
                    parents=(claim_uri, candidate_uri),
                    summary="unverified proposed witness",
                )
                witness_uri = stored_witness.artifact_uri

            return WitnessFindResult(
                status=response.status,
                result=ResultEnvelope(
                    execution=Execution(
                        status=ExecutionStatus.COMPLETED,
                        runtime_ms=execution.runtime_ms,
                    ),
                    input=InputValidation(status=InputStatus.ACCEPTED),
                    conclusion=_proposed_conclusion(
                        response.status,
                        response.role,
                    ),
                    assurance=Assurance(
                        arithmetic=response.arithmetic,
                        method=(
                            Method.DIRECT_WITNESS
                            if response.status == WitnessSearchStatus.FOUND
                            else Method.BOUNDED_SEARCH
                        ),
                        coverage=(
                            Coverage.NOT_APPLICABLE
                            if response.status == WitnessSearchStatus.FOUND
                            else response.coverage
                        ),
                        verification=Verification.UNVERIFIED,
                    ),
                    claim_digest=claim.manifest.object_digest,
                    semantics_digest=semantics.manifest.object_digest,
                    candidate_digest=candidate.manifest.object_digest,
                    evidence_uris=(witness_uri,) if witness_uri else (),
                ),
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                plugin_id=plugin_id,
                witness_uri=witness_uri,
                detail=response.detail,
            )
        except (
            StorageError,
            SchemaRegistryError,
            PluginRegistryError,
            ValidationError,
            ValueError,
        ) as exc:
            detail = _witness_failure_detail(exc)
            return self._rejected(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                plugin_id=plugin_id,
                detail=detail,
            )

    @staticmethod
    def _rejected(
        *,
        claim_uri: str,
        candidate_uri: str,
        plugin_id: str,
        detail: str,
    ) -> WitnessFindResult:
        return WitnessFindResult(
            status=WitnessSearchStatus.UNKNOWN,
            result=ResultEnvelope(
                execution=Execution(status=ExecutionStatus.COMPLETED),
                input=InputValidation(
                    status=InputStatus.REJECTED,
                    errors=(detail,),
                ),
                conclusion=Conclusion.UNKNOWN,
                assurance=Assurance(
                    arithmetic=Arithmetic.SYMBOLIC,
                    method=Method.HEURISTIC,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
            ),
            claim_uri=claim_uri,
            candidate_uri=candidate_uri,
            plugin_id=plugin_id,
            detail=detail,
        )


def _proposed_conclusion(
    status: WitnessSearchStatus,
    role: WitnessRole | None,
) -> Conclusion:
    if status != WitnessSearchStatus.FOUND or role is None:
        return Conclusion.UNKNOWN
    if role in {
        WitnessRole.DEFEATS_CANDIDATE,
        WitnessRole.REFUTES_CLAIM,
    }:
        return Conclusion.FALSE
    return Conclusion.TRUE


def _witness_failure_detail(exc: Exception) -> str:
    if isinstance(exc, ValueError) and not isinstance(
        exc,
        (PluginRegistryError, SchemaRegistryError, StorageError, ValidationError),
    ):
        return str(exc)
    _LOGGER.warning("witness search failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "A required claim or candidate artifact is unavailable. Check the "
            "artifact URIs, then retry."
        )
    if isinstance(exc, PluginRegistryError):
        return (
            "The witness plugin is unavailable. Call math.find, choose "
            "an installed reference domain, and retry."
        )
    return (
        "The claim, candidate, or plugin response is invalid. Check the reference "
        "contract and retry with matching artifacts."
    )
