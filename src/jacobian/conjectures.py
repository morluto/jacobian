"""Hypothesis-producing workflows built on the durable search boundary."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.claims import ClaimValidationService
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.conjectures import (
    ConjectureOperation,
    ConjectureWorkflowRequest,
    ConjectureWorkflowResult,
    HypothesisRecord,
    HypothesisTransformationRecord,
    NoveltyAssessment,
    ParameterRegion,
    ParameterRegionEvidence,
    ParameterRegionKind,
    ParameterRegionSubject,
    PluginHypothesisResponse,
)
from jacobian.contracts.evidence import WitnessEnvelope, WitnessRole
from jacobian.contracts.plugins import CapabilityName, PluginManifest
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Verification,
)
from jacobian.contracts.search import SearchRunRequest
from jacobian.contracts.verification import VerificationRecord
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import (
    PluginRegistry,
    PluginRegistryError,
    ResolvedCapability,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.search import SearchError, SearchService
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

_LOGGER = logging.getLogger(__name__)


class ConjectureError(RuntimeError):
    """A conjecture workflow request or plugin response is invalid."""


class ConjectureService:
    """Create unverified hypotheses and optionally route them through search.

    Hypothesis plugins own domain grammar and heuristics. This service owns
    exact source/evidence replay, immutable edit lineage, schema validation,
    deduplication, and the rule that plugin output cannot promote evidence.
    """

    def __init__(
        self,
        store: ArtifactRepository,
        schemas: SchemaRegistry,
        plugins: PluginRegistry,
        claims: ClaimValidationService,
        executor: PluginExecutor,
        search: SearchService,
        verification: VerificationService,
    ) -> None:
        self.store = store
        self.schemas = schemas
        self.plugins = plugins
        self.claims = claims
        self.executor = executor
        self.search = search
        self.verification = verification
        self.semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.hypothesis-transformation",
            version="1",
            definition={
                "description": ("unverified conjecture edits and exact source lineage")
            },
        )
        self.transformation_schema_uri = schemas.register(
            name="jacobian.hypothesis-transformation",
            version="1",
            schema=model_schema(HypothesisTransformationRecord),
        )
        self.parameter_region_subject_schema_uri = schemas.register(
            name="jacobian.parameter-region-subject",
            version="1",
            schema=model_schema(ParameterRegionSubject),
        )

    def run(
        self,
        request: ConjectureWorkflowRequest | dict[str, Any],
    ) -> ConjectureWorkflowResult:
        """Run one bounded hypothesis transformation and optional falsification."""

        started = time.monotonic()
        try:
            selected = ConjectureWorkflowRequest.model_validate(request)
        except ValidationError as exc:
            return self._rejected(
                operation=None,
                plugin_id=None,
                request_digest=_digest_untrusted_request(request),
                detail=_model_validation_detail(exc, "conjecture request"),
                started=started,
            )
        request_digest = _digest(selected.model_dump(mode="json"))
        try:
            manifest, capability, source, evidence = self._prepare(selected)
        except (
            ConjectureError,
            PluginRegistryError,
            SchemaRegistryError,
            StorageError,
            ValidationError,
            ValueError,
        ) as exc:
            return self._rejected(
                operation=selected.operation,
                plugin_id=selected.plugin_id,
                request_digest=request_digest,
                detail=_workflow_failure_detail(exc),
                started=started,
            )

        plugin_request = {
            "request_version": "1",
            "operation": selected.operation.value,
            "source": (_artifact_view(source) if source is not None else None),
            "evidence": [_artifact_view(artifact) for artifact in evidence],
            "constraints": selected.constraints,
            "reference_claim_uris": list(selected.reference_claim_uris),
            "seed": selected.seed,
            "max_hypotheses": selected.max_hypotheses,
            "bindings": {
                "plugin_id": selected.plugin_id,
                "registry_snapshot_uri": capability.registry_snapshot_uri,
                "implementation_digest": capability.implementation_digest,
                "request_digest": request_digest,
            },
        }
        execution = self.executor.run(
            entrypoint=capability.descriptor.entrypoint,
            implementation_digest=capability.implementation_digest,
            request=plugin_request,
            timeout_seconds=selected.wall_seconds,
        )
        if execution.status is not ExecutionStatus.COMPLETED:
            return ConjectureWorkflowResult(
                operation=selected.operation,
                execution=Execution(
                    status=execution.status,
                    runtime_ms=execution.runtime_ms,
                    detail=execution.detail,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                request_digest=request_digest,
                plugin_id=selected.plugin_id,
                registry_snapshot_uri=capability.registry_snapshot_uri,
                implementation_digest=capability.implementation_digest,
                detail=execution.detail or "hypothesis transformer failed",
            )
        try:
            response = PluginHypothesisResponse.model_validate(execution.output)
            if len(response.proposals) > selected.max_hypotheses:
                raise ConjectureError(
                    "The plugin returned more hypotheses than max_hypotheses. "
                    "Reduce the plugin output or raise that limit, then retry."
                )
            hypotheses = self._commit_hypotheses(
                selected=selected,
                request_digest=request_digest,
                manifest=manifest,
                capability=capability,
                source=source,
                evidence=evidence,
                response=response,
            )
        except (
            ConjectureError,
            PluginRegistryError,
            SchemaRegistryError,
            SearchError,
            StorageError,
            ValidationError,
            ValueError,
        ) as exc:
            detail = _workflow_failure_detail(exc)
            return ConjectureWorkflowResult(
                operation=selected.operation,
                execution=Execution(
                    status=ExecutionStatus.ERROR,
                    runtime_ms=_elapsed_ms(started),
                    detail=detail,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                request_digest=request_digest,
                plugin_id=selected.plugin_id,
                registry_snapshot_uri=capability.registry_snapshot_uri,
                implementation_digest=capability.implementation_digest,
                detail=detail,
            )
        return ConjectureWorkflowResult(
            operation=selected.operation,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_elapsed_ms(started),
            ),
            input=InputValidation(status=InputStatus.ACCEPTED),
            request_digest=request_digest,
            plugin_id=selected.plugin_id,
            registry_snapshot_uri=capability.registry_snapshot_uri,
            implementation_digest=capability.implementation_digest,
            hypotheses=hypotheses,
            verification=Verification.UNVERIFIED,
            detail=response.detail,
        )

    def promote_parameter_region(
        self,
        *,
        subject_uri: str,
        verification_record_uri: str,
    ) -> ParameterRegion:
        """Replay an exact certificate record before promoting a region.

        Promotion requires both object-digest bindings and the exact subject
        and claim artifact URIs in the verification record's parents. The URI
        checks prevent substitution of an equal payload carried by different
        lineage metadata. Replay must reproduce the supplied record URI, which
        binds the authorized checker, evidence, request, and environment.
        """

        try:
            subject_artifact = self.store.get(subject_uri)
            if (
                subject_artifact.manifest.schema_uri
                != self.parameter_region_subject_schema_uri
            ):
                raise ConjectureError(
                    "This parameter region uses the wrong schema. Promote the "
                    "subject_uri returned by parameter.generalize, then retry."
                )
            subject = ParameterRegionSubject.model_validate(subject_artifact.payload)
            claim = self.store.get(subject.claim_uri)
            subject_parents = set(subject_artifact.manifest.parents)
            if subject.claim_uri not in subject_parents or not set(
                subject.sample_uris
            ).issubset(subject_parents):
                raise ConjectureError(
                    "This parameter region is missing its claim or sample lineage. "
                    "Run parameter.generalize again and promote the returned subject."
                )
            if claim.manifest.semantics_uri != subject_artifact.manifest.semantics_uri:
                raise ConjectureError(
                    "The parameter region and target claim use different semantics. "
                    "Use artifacts from one reference contract, then retry."
                )
            record_artifact = self.store.get(verification_record_uri)
            record = VerificationRecord.model_validate(record_artifact.payload)
            record_parents = set(record_artifact.manifest.parents)
            if subject_artifact.artifact_uri not in record_parents:
                raise ConjectureError(
                    "The verification record does not cover this parameter region. "
                    "Verify the supplied subject_uri, then retry."
                )
            if subject.claim_uri not in record_parents:
                raise ConjectureError(
                    "The verification record does not cover this region's claim. "
                    "Verify the exact claim and subject together, then retry."
                )
            if record.evidence_kind is not EvidenceKind.CERTIFICATE:
                raise ConjectureError(
                    "Parameter-region promotion requires certificate evidence. "
                    "Run certificate verification for this subject, then retry."
                )
            if record.conclusion is not Conclusion.TRUE:
                raise ConjectureError(
                    "The certificate did not establish the parameter region. "
                    "Provide a certificate with conclusion TRUE, then retry."
                )
            if record.bindings.claim_digest != claim.manifest.object_digest:
                raise ConjectureError(
                    "The verification record covers a different claim. Verify this "
                    "region's exact claim, then retry."
                )
            if (
                record.bindings.candidate_digest
                != subject_artifact.manifest.object_digest
            ):
                raise ConjectureError(
                    "The verification record covers different parameter conditions. "
                    "Verify this exact subject_uri, then retry."
                )
            self._replay_verification_record(
                source=subject_artifact,
                record_artifact=record_artifact,
                record=record,
            )
        except ConjectureError:
            raise
        except (StorageError, ValidationError, ValueError) as exc:
            raise ConjectureError(_workflow_failure_detail(exc)) from exc

        evidence = (
            ParameterRegionEvidence.VERIFIED_SUFFICIENT
            if subject.kind is ParameterRegionKind.SUFFICIENT
            else ParameterRegionEvidence.VERIFIED_NECESSARY
        )
        return ParameterRegion(
            kind=subject.kind,
            conditions=subject.conditions,
            evidence=evidence,
            sample_uris=subject.sample_uris,
            subject_uri=subject_artifact.artifact_uri,
            verification_record_uri=record_artifact.artifact_uri,
        )

    def _prepare(
        self,
        request: ConjectureWorkflowRequest,
    ) -> tuple[
        PluginManifest,
        ResolvedCapability,
        StoredArtifact | None,
        tuple[StoredArtifact, ...],
    ]:
        manifest = self.plugins.get(request.plugin_id)
        capability = self.plugins.resolve(
            request.plugin_id,
            CapabilityName.HYPOTHESIS_TRANSFORMER,
        )
        source = (
            self.store.get(request.source_uri)
            if request.source_uri is not None
            else None
        )
        if source is not None and (
            source.manifest.semantics_uri != manifest.semantics_uri
        ):
            raise ConjectureError(
                "The source and hypothesis plugin use different semantics. Choose "
                "the plugin from the source's reference contract, then retry."
            )
        evidence: list[StoredArtifact] = []
        if request.verification_record_uri is not None:
            record_artifact = self.store.get(request.verification_record_uri)
            self._validate_verified_source(
                request=request,
                source=source,
                record_artifact=record_artifact,
            )
            evidence.append(record_artifact)
            record = VerificationRecord.model_validate(record_artifact.payload)
            evidence.append(self.store.get(record.evidence_uri))
        for claim_uri in request.reference_claim_uris:
            reference = self.store.get(claim_uri)
            if (
                reference.manifest.schema_uri != manifest.claim_schema_uri
                or reference.manifest.semantics_uri != manifest.semantics_uri
            ):
                raise ConjectureError(
                    "A reference claim does not match the plugin's claim contract. "
                    "Use claims returned by the same reference domain, then retry."
                )
            self.schemas.validate(reference.manifest.schema_uri, reference.payload)
        known_evidence_uris = {artifact.artifact_uri for artifact in evidence}
        for evidence_uri in request.evidence_uris:
            if evidence_uri not in known_evidence_uris:
                evidence.append(self.store.get(evidence_uri))
                known_evidence_uris.add(evidence_uri)
        if request.operation is ConjectureOperation.REPAIR:
            if source is None:
                raise ConjectureError(
                    "Conjecture repair requires source_uri. Supply the claim to repair, "
                    "then retry."
                )
            if source.manifest.schema_uri != manifest.claim_schema_uri:
                raise ConjectureError(
                    "The repair source does not match the plugin's claim schema. "
                    "Choose the plugin from the source's reference contract."
                )
            validation = self.claims.validate(
                claim_uri=source.artifact_uri,
                plugin_id=request.plugin_id,
            )
            if not validation.valid:
                raise ConjectureError(
                    "The repair source is not a valid claim for this plugin. Check "
                    "claim.validate, correct the source, then retry. Validation: "
                    + "; ".join(validation.input.errors)
                )
        if request.falsification is not None:
            search_capabilities = [
                self.plugins.resolve(request.plugin_id, CapabilityName.PROPOSER),
                self.plugins.resolve(request.plugin_id, CapabilityName.REFINER),
                self.plugins.resolve(request.plugin_id, CapabilityName.EVALUATOR),
            ]
            if request.falsification.witness_role is not None:
                search_capabilities.append(
                    self.plugins.resolve(
                        request.plugin_id,
                        CapabilityName.WITNESS_ORACLE,
                    )
                )
            if {
                resolved.registry_snapshot_uri
                for resolved in (capability, *search_capabilities)
            } != {capability.registry_snapshot_uri}:
                raise ConjectureError(
                    "The hypothesis and falsification capabilities were loaded from "
                    "different plugin versions. Reload Jacobian, then retry."
                )
        return manifest, capability, source, tuple(evidence)

    def _validate_verified_source(
        self,
        *,
        request: ConjectureWorkflowRequest,
        source: StoredArtifact | None,
        record_artifact: StoredArtifact,
    ) -> None:
        if source is None:
            raise ConjectureError(
                "This verification record requires source_uri. Supply the exact "
                "verified source artifact, then retry."
            )
        if record_artifact.manifest.schema_uri != self.verification.record_schema_uri:
            raise ConjectureError(
                "The supplied record is not a Jacobian verification record. Use the "
                "verification_record_uri returned by a verification tool."
            )
        record = VerificationRecord.model_validate(record_artifact.payload)
        if source.artifact_uri not in record_artifact.manifest.parents:
            raise ConjectureError(
                "The verification record covers a different source. Supply the exact "
                "source used during verification, then retry."
            )
        self._replay_verification_record(
            source=source,
            record_artifact=record_artifact,
            record=record,
        )
        source_digest = source.manifest.object_digest
        semantics_digest = self.store.get(
            source.manifest.semantics_uri
        ).manifest.object_digest
        if record.bindings.semantics_digest != semantics_digest:
            raise ConjectureError(
                "The verification record and source use different semantics. Supply "
                "the exact verified source, then retry."
            )
        if request.operation is ConjectureOperation.REPAIR:
            witness = WitnessEnvelope.model_validate(
                self.store.get(record.evidence_uri).payload
            )
            if (
                record.evidence_kind is not EvidenceKind.WITNESS
                or record.conclusion is not Conclusion.FALSE
                or record.bindings.claim_digest != source_digest
                or witness.role is not WitnessRole.REFUTES_CLAIM
            ):
                raise ConjectureError(
                    "Repair requires a verified counterexample for the source claim. "
                    "Run witness verification with role REFUTES_CLAIM, then retry."
                )
            return
        manifest = self.plugins.get(request.plugin_id)
        if (
            source.manifest.schema_uri != manifest.candidate_schema_uri
            or record.evidence_kind
            not in {EvidenceKind.WITNESS, EvidenceKind.CERTIFICATE}
            or record.conclusion is not Conclusion.TRUE
            or record.bindings.candidate_digest != source_digest
        ):
            raise ConjectureError(
                "Parameter generalization requires a verified construction candidate. "
                "Verify the source candidate first, then retry."
            )
        if record.evidence_kind is EvidenceKind.WITNESS:
            witness = WitnessEnvelope.model_validate(
                self.store.get(record.evidence_uri).payload
            )
            if witness.role is not WitnessRole.RESCUES_CANDIDATE:
                raise ConjectureError(
                    "The verification witness does not establish this construction. "
                    "Use a verified RESCUES_CANDIDATE witness, then retry."
                )

    def _replay_verification_record(
        self,
        *,
        source: StoredArtifact,
        record_artifact: StoredArtifact,
        record: VerificationRecord,
    ) -> None:
        if record.evidence_kind is EvidenceKind.WITNESS:
            claim_uri = (
                source.artifact_uri
                if record.bindings.claim_digest == source.manifest.object_digest
                else self._record_parent_for_digest(
                    record_artifact,
                    record.bindings.claim_digest,
                    label="claim",
                )
            )
            candidate_digest = record.bindings.candidate_digest
            if candidate_digest is None:
                raise ConjectureError(
                    "The witness verification record does not identify a candidate. "
                    "Re-run verification for the exact claim and candidate."
                )
            candidate_uri = (
                source.artifact_uri
                if candidate_digest == source.manifest.object_digest
                else self._record_parent_for_digest(
                    record_artifact,
                    candidate_digest,
                    label="candidate",
                )
            )
            replay = self.verification.verify_witness(
                claim_uri=claim_uri,
                candidate_uri=candidate_uri,
                witness_uri=record.evidence_uri,
                checker_id=record.checker_id,
            )
        elif record.evidence_kind is EvidenceKind.CERTIFICATE:
            replay = self.verification.verify_certificate(
                certificate_uri=record.evidence_uri,
                checker_id=record.checker_id,
            )
        else:
            raise ConjectureError(
                "This workflow cannot use the record's evidence type. Use a verified "
                "witness or certificate, then retry."
            )
        if (
            replay.assurance.verification is not Verification.VERIFIED
            or replay.verification_record_uri != record_artifact.artifact_uri
        ):
            raise ConjectureError(
                "The verification record no longer replays to VERIFIED. Re-run "
                "verification with an active checker, then retry this workflow."
            )

    def _record_parent_for_digest(
        self,
        record_artifact: StoredArtifact,
        object_digest: str,
        *,
        label: str,
    ) -> str:
        parent_uris = set(record_artifact.manifest.parents)
        matches = [
            uri
            for uri in self.store.find_by_object_digest(object_digest)
            if uri in parent_uris
        ]
        if len(matches) != 1:
            raise ConjectureError(
                f"The verification record does not identify exactly one {label}. "
                "Re-run verification from the exact input artifacts, then retry."
            )
        return matches[0]

    def _commit_hypotheses(
        self,
        *,
        selected: ConjectureWorkflowRequest,
        request_digest: str,
        manifest: PluginManifest,
        capability: ResolvedCapability,
        source: StoredArtifact | None,
        evidence: tuple[StoredArtifact, ...],
        response: PluginHypothesisResponse,
    ) -> tuple[HypothesisRecord, ...]:
        reference_digests = {
            self.store.get(uri).manifest.object_digest
            for uri in selected.reference_claim_uris
        }
        authorized_sample_uris = {artifact.artifact_uri for artifact in evidence}
        seen_digests: set[str] = set()
        records: list[HypothesisRecord] = []
        for proposal in response.proposals:
            for sample_uri in (
                proposal.parameter_region.sample_uris
                if proposal.parameter_region is not None
                else ()
            ):
                if sample_uri not in authorized_sample_uris:
                    raise ConjectureError(
                        "A parameter-region sample was not supplied as workflow "
                        "evidence. Add its artifact URI to evidence_uris, then retry."
                    )
            normalized_claim = self.schemas.validate(
                manifest.claim_schema_uri,
                proposal.claim,
            )
            parents = (
                *((source.artifact_uri,) if source is not None else ()),
                *(artifact.artifact_uri for artifact in evidence),
                selected.plugin_id,
                capability.registry_snapshot_uri,
            )
            claim = self.store.put(
                schema_uri=manifest.claim_schema_uri,
                semantics_uri=manifest.semantics_uri,
                payload=normalized_claim,
                parents=parents,
                summary="unverified generated hypothesis",
            )
            if (
                claim.object_digest in seen_digests
                or claim.object_digest in reference_digests
            ):
                continue
            validation = self.claims.validate(
                claim_uri=claim.artifact_uri,
                plugin_id=selected.plugin_id,
            )
            if not validation.valid:
                raise ConjectureError(
                    "The plugin generated an invalid claim. Check its output against "
                    "math.find, then retry. Validation: "
                    + "; ".join(validation.input.errors)
                )
            seen_digests.add(claim.object_digest)
            committed_region = proposal.parameter_region
            if committed_region is not None:
                subject = ParameterRegionSubject(
                    claim_uri=claim.artifact_uri,
                    kind=committed_region.kind,
                    conditions=committed_region.conditions,
                    sample_uris=committed_region.sample_uris,
                )
                stored_subject = self.store.put(
                    schema_uri=self.parameter_region_subject_schema_uri,
                    semantics_uri=manifest.semantics_uri,
                    payload=self.schemas.validate(
                        self.parameter_region_subject_schema_uri,
                        subject.model_dump(mode="json"),
                    ),
                    parents=_deduplicate_uris(
                        (
                            claim.artifact_uri,
                            selected.plugin_id,
                            capability.registry_snapshot_uri,
                            *committed_region.sample_uris,
                        )
                    ),
                    summary="immutable parameter-region verification subject",
                )
                committed_region = committed_region.model_copy(
                    update={"subject_uri": stored_subject.artifact_uri}
                )
            transformation = HypothesisTransformationRecord(
                operation=selected.operation,
                source_uri=source.artifact_uri if source is not None else None,
                target_claim_uri=claim.artifact_uri,
                edit=proposal.edit,
                metrics=proposal.metrics,
                parameter_region=committed_region,
                evidence_uris=tuple(artifact.artifact_uri for artifact in evidence),
                plugin_id=selected.plugin_id,
                registry_snapshot_uri=capability.registry_snapshot_uri,
                implementation_digest=capability.implementation_digest,
                request_digest=request_digest,
            )
            transformation_parents = (
                claim.artifact_uri,
                *parents,
                *(committed_region.sample_uris if committed_region is not None else ()),
                *(
                    (committed_region.subject_uri,)
                    if committed_region is not None
                    else ()
                ),
            )
            stored_transformation = self.store.put(
                schema_uri=self.transformation_schema_uri,
                semantics_uri=self.semantics_uri,
                payload=self.schemas.validate(
                    self.transformation_schema_uri,
                    transformation.model_dump(mode="json"),
                ),
                parents=_deduplicate_uris(transformation_parents),
                summary="hypothesis transformation lineage",
            )
            record = HypothesisRecord(
                claim_uri=claim.artifact_uri,
                transformation_uri=stored_transformation.artifact_uri,
                novelty=NoveltyAssessment.UNKNOWN,
                parameter_region=committed_region,
                detail=proposal.detail,
            )
            if selected.falsification is not None:
                record = self._falsify(
                    selected=selected,
                    request_digest=request_digest,
                    record=record,
                )
            records.append(record)
        return tuple(records)

    def _falsify(
        self,
        *,
        selected: ConjectureWorkflowRequest,
        request_digest: str,
        record: HypothesisRecord,
    ) -> HypothesisRecord:
        plan = selected.falsification
        if plan is None:
            raise ConjectureError(
                "Falsification requires a plan. Supply falsification settings or omit "
                "falsification, then retry."
            )
        idempotency_digest = hashlib.sha256(
            canonicalize_json(
                {
                    "workflow_request_digest": request_digest,
                    "claim_uri": record.claim_uri,
                    "operation": selected.operation.value,
                }
            )
        ).hexdigest()
        handle = self.search.start(
            SearchRunRequest(
                idempotency_key=f"hypothesis:{idempotency_digest}",
                claim_uri=record.claim_uri,
                plugin_id=selected.plugin_id,
                initial_state=plan.initial_state,
                seed=selected.seed,
                witness_role=plan.witness_role,
                counterexample_checker_id=plan.counterexample_checker_id,
                budget=plan.budget,
            )
        )
        try:
            snapshot = self.search.wait(
                handle.experiment_uri,
                timeout_seconds=plan.budget.wall_seconds + 5,
            )
        except TimeoutError:
            self.search.cancel(handle.experiment_uri)
            try:
                snapshot = self.search.wait(
                    handle.experiment_uri,
                    timeout_seconds=5,
                )
            except TimeoutError:
                snapshot = self.search.inspect(handle.experiment_uri)
        return HypothesisRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "search_experiment_uri": snapshot.experiment_uri,
                "verified_counterexamples": (
                    snapshot.accounting.verified_counterexamples
                ),
                "detail": (
                    f"{record.detail}; falsification={snapshot.state.value}"
                ).strip("; "),
            }
        )

    @staticmethod
    def _rejected(
        *,
        operation: ConjectureOperation | None,
        plugin_id: str | None,
        request_digest: str,
        detail: str,
        started: float,
    ) -> ConjectureWorkflowResult:
        return ConjectureWorkflowResult(
            operation=operation,
            execution=Execution(
                status=ExecutionStatus.COMPLETED,
                runtime_ms=_elapsed_ms(started),
            ),
            input=InputValidation(
                status=InputStatus.REJECTED,
                errors=(detail,),
            ),
            request_digest=request_digest,
            plugin_id=plugin_id,
            detail=detail,
        )


def _artifact_view(artifact: StoredArtifact) -> dict[str, Any]:
    return {
        "artifact_uri": artifact.artifact_uri,
        "object_digest": artifact.manifest.object_digest,
        "schema_uri": artifact.manifest.schema_uri,
        "semantics_uri": artifact.manifest.semantics_uri,
        "payload": artifact.payload,
    }


def _workflow_failure_detail(exc: Exception) -> str:
    if isinstance(exc, ConjectureError):
        return str(exc)
    _LOGGER.warning("conjecture workflow failed", exc_info=exc)
    if isinstance(exc, StorageError):
        return (
            "Jacobian could not read or save conjecture workflow data. Check the "
            "state directory and available disk space, then retry."
        )
    if isinstance(exc, (PluginRegistryError, SchemaRegistryError)):
        return (
            "The selected plugin or reference contract is unavailable. Call "
            "math.find, choose an installed option, and retry."
        )
    if isinstance(exc, SearchError):
        return (
            "The falsification search could not continue. Inspect the returned "
            "experiment state, correct the request, and retry."
        )
    if isinstance(exc, ValidationError):
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        if any(error["type"] == "parameter_region_promotion" for error in errors):
            return (
                "The plugin tried to promote parameter-region evidence. Plugin "
                "output must remain unverified; remove the promoted evidence field, "
                "then retry."
            )
    return (
        "The conjecture workflow received invalid data. Check the request and plugin "
        "output against math.find, then retry."
    )


def _model_validation_detail(exc: ValidationError, subject: str) -> str:
    first = exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[0]
    path = ".".join(str(part) for part in first["loc"]) or "request"
    return (
        f"The {subject} is invalid at {path}: {first['msg']}. Check "
        "math.find, correct that field, and retry."
    )


def _digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _deduplicate_uris(uris: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(uris))


def _digest_untrusted_request(
    request: ConjectureWorkflowRequest | dict[str, Any],
) -> str:
    try:
        payload = (
            request.model_dump(mode="json")
            if isinstance(request, ConjectureWorkflowRequest)
            else request
        )
        return _digest(payload)
    except ValueError:
        return _digest({"invalid_request": True})
