"""Finite case partition capability with independent replay."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityMode,
    CapabilityObligation,
    CapabilityObligationStatus,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.results import (
    Execution,
    ExecutionStatus,
    ResultEnvelope,
    Verification,
)
from jacobian.domains._examples import example
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

_ARTIFACT_PATTERN = r"^artifact://sha256/[0-9a-f]{64}$"


@dataclass(frozen=True, slots=True)
class FinitePartitionInstallation:
    checker_id: str | None
    semantics_uri: str
    scope_schema_uri: str
    claim_schema_uri: str
    partition_schema_uri: str
    certificate_schema_uri: str


def install_finite_partition(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[FinitePartitionAdapter, FinitePartitionInstallation]:
    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.finite-enumerated-partition",
        version="1",
        definition={
            "domain": "finite sets of distinct string identifiers",
            "partition": "named cases whose union is the scope",
            "disjointness": "required when require_disjoint is true",
        },
    )
    scope_schema_uri = schemas.register(
        name="jacobian.finite-enumerated-scope",
        version="1",
        schema={
            "type": "object",
            "properties": {
                "scope_schema_version": {"const": "1"},
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
            "required": ["scope_schema_version", "elements"],
            "additionalProperties": False,
        },
    )
    claim_schema_uri = schemas.register(
        name="jacobian.finite-partition-claim",
        version="1",
        schema={
            "type": "object",
            "properties": {
                "claim_schema_version": {"const": "1"},
                "predicate": {"const": "finite_partition"},
                "require_disjoint": {"type": "boolean"},
            },
            "required": [
                "claim_schema_version",
                "predicate",
                "require_disjoint",
            ],
            "additionalProperties": False,
        },
    )
    partition_schema_uri = schemas.register(
        name="jacobian.finite-partition",
        version="1",
        schema={
            "type": "object",
            "properties": {
                "partition_schema_version": {"const": "1"},
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string", "minLength": 1},
                            "members": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                        },
                        "required": ["case_id", "members"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["partition_schema_version", "cases"],
            "additionalProperties": False,
        },
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    registration = CheckerInstaller(checkers).install(
        CheckerOperation(
            name="finite partition coverage checker",
            entrypoint="jacobian_checkers.finite_partition:check_partition",
            evidence_kind=EvidenceKind.CERTIFICATE,
            format_id="finite.partition",
            format_version="1",
            claim_schema_uris=(claim_schema_uri,),
            semantics_uris=(semantics_uri,),
            candidate_schema_uris=(partition_schema_uri,),
            reason="operator requested bundled reference checker installation",
        ),
        authorize=authorize_checker,
    )
    checker_id = registration.checker_id
    installation = FinitePartitionInstallation(
        checker_id=checker_id,
        semantics_uri=semantics_uri,
        scope_schema_uri=scope_schema_uri,
        claim_schema_uri=claim_schema_uri,
        partition_schema_uri=partition_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
    )
    return (
        FinitePartitionAdapter(artifacts, store, verification, installation),
        installation,
    )


class FinitePartitionAdapter:
    """Propose or independently verify a finite partition."""

    def __init__(
        self,
        artifacts: ArtifactService,
        store: ArtifactRepository,
        verification: VerificationService,
        installation: FinitePartitionInstallation,
    ) -> None:
        self.artifacts = artifacts
        self.store = store
        self.verification = verification
        self.installation = installation
        modes = (
            (CapabilityMode.EXPLORE, CapabilityMode.VERIFY)
            if installation.checker_id is not None
            else (CapabilityMode.EXPLORE,)
        )
        self._descriptor = CapabilityDescriptor(
            capability_id="case.partition.finite",
            version="1",
            title="Partition an explicit finite domain",
            description=(
                "Materialize named cases over an explicit finite scope and optionally "
                "replay exact coverage and disjointness with an authorized checker. "
                "Members and case labels are opaque caller-supplied strings; the "
                "checker does not establish their mathematical meaning or that the "
                "supplied universe exhausts an external domain."
            ),
            provider="jacobian.finite",
            provider_runtime=known_provider_runtime(
                "jacobian.finite",
                features=("finite-partition",),
                checker_ids=(
                    (installation.checker_id,)
                    if installation.checker_id is not None
                    else ()
                ),
            ),
            modes=modes,
            input_schema={
                "type": "object",
                "properties": {
                    "universe": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "case_id": {"type": "string", "minLength": 1},
                                "members": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "uniqueItems": True,
                                },
                            },
                            "required": ["case_id", "members"],
                            "additionalProperties": False,
                        },
                    },
                    "require_disjoint": {"type": "boolean"},
                },
                "required": ["universe", "cases"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scope_uri": {"type": "string", "pattern": _ARTIFACT_PATTERN},
                    "claim_uri": {"type": "string", "pattern": _ARTIFACT_PATTERN},
                    "partition_uri": {"type": "string", "pattern": _ARTIFACT_PATTERN},
                    "certificate_uri": {
                        "type": ["string", "null"],
                        "pattern": _ARTIFACT_PATTERN,
                    },
                    "verification_record_uri": {
                        "type": ["string", "null"],
                        "pattern": _ARTIFACT_PATTERN,
                    },
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "outside": {"type": "array", "items": {"type": "string"}},
                    "overlaps": {"type": "array", "items": {"type": "string"}},
                    "duplicate_case_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "scope_uri",
                    "claim_uri",
                    "partition_uri",
                    "certificate_uri",
                    "verification_record_uri",
                    "missing",
                    "outside",
                    "overlaps",
                    "duplicate_case_ids",
                ],
                "additionalProperties": False,
            },
            tags=("cases", "finite", "coverage", "verification"),
            invocation_examples=(
                example(
                    "singleton_partition",
                    "Partition a singleton universe into one case.",
                    {
                        "universe": ["a"],
                        "cases": [{"case_id": "all", "members": ["a"]}],
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        universe = [str(item) for item in request.input["universe"]]
        cases = [
            {
                "case_id": str(case["case_id"]),
                "members": [str(member) for member in case["members"]],
            }
            for case in request.input["cases"]
        ]
        require_disjoint = bool(request.input.get("require_disjoint", True))
        scope = self.artifacts.put(
            schema_uri=self.installation.scope_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload={"scope_schema_version": "1", "elements": universe},
            summary="explicit finite case scope",
        )
        claim = self.artifacts.put(
            schema_uri=self.installation.claim_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload={
                "claim_schema_version": "1",
                "predicate": "finite_partition",
                "require_disjoint": require_disjoint,
            },
            parents=(scope.artifact_uri,),
            summary="finite partition coverage obligation",
        )
        partition = self.artifacts.put(
            schema_uri=self.installation.partition_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload={"partition_schema_version": "1", "cases": cases},
            parents=(scope.artifact_uri, claim.artifact_uri),
            summary="proposed finite partition",
        )
        missing, outside, overlaps, duplicate_case_ids = _partition_diagnostics(
            universe, cases
        )
        certificate_uri = None
        record_uri = None
        verified = False
        verification_result: ResultEnvelope | None = None
        artifact_uris = [scope.artifact_uri, claim.artifact_uri, partition.artifact_uri]
        if request.mode is CapabilityMode.VERIFY:
            certificate_uri, verification_result = self._verify(
                scope_uri=scope.artifact_uri,
                claim_uri=claim.artifact_uri,
                partition_uri=partition.artifact_uri,
            )
            record_uri = verification_result.verification_record_uri
            verified = (
                verification_result.assurance.verification is Verification.VERIFIED
            )
            artifact_uris.append(certificate_uri)
            if record_uri is not None:
                artifact_uris.append(record_uri)
        assurance_level = (
            CapabilityAssuranceLevel.VERIFIED
            if verified
            else CapabilityAssuranceLevel.COMPUTED
        )
        relationship_status = (
            CapabilityRelationshipStatus.VERIFIED
            if verified
            else CapabilityRelationshipStatus.PROPOSED
        )
        obligation_status = (
            CapabilityObligationStatus.DISCHARGED
            if verified
            else CapabilityObligationStatus.OPEN
        )
        execution_status = (
            verification_result.execution.status
            if verification_result is not None
            else ExecutionStatus.COMPLETED
        )
        complete = (
            execution_status is ExecutionStatus.COMPLETED
            and not missing
            and not outside
            and not duplicate_case_ids
            and (not require_disjoint or not overlaps)
        )
        verified_replay_basis = (
            "authorized checker replayed equality-based coverage and required "
            "disjointness within the caller-supplied universe"
            if require_disjoint
            else (
                "authorized checker replayed equality-based coverage within the "
                "caller-supplied universe; disjointness was not required"
            )
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=execution_status,
                runtime_ms=int((time.monotonic() - started) * 1000),
                detail=(
                    verification_result.execution.detail
                    if verification_result is not None
                    else None
                ),
            ),
            output={
                "scope_uri": scope.artifact_uri,
                "claim_uri": claim.artifact_uri,
                "partition_uri": partition.artifact_uri,
                "certificate_uri": certificate_uri,
                "verification_record_uri": record_uri,
                "missing": missing,
                "outside": outside,
                "overlaps": overlaps,
                "duplicate_case_ids": duplicate_case_ids,
            },
            scope=CapabilityScope(
                description=(
                    "the exact caller-supplied finite universe; external-domain "
                    "completeness and member semantics are not checked"
                ),
                parameters={"element_count": len(universe)},
                artifact_uri=scope.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if complete
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=(
                    f"{verified_replay_basis}; it did not check external-domain "
                    "completeness or member/case semantics"
                    if verified
                    else "generator-side membership accounting; not independently checked"
                ),
                assurance_level=assurance_level,
                verification_record_uri=record_uri if verified else None,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="case.relation.partitions",
                    source_artifact_uris=(scope.artifact_uri,),
                    target_artifact_uris=(partition.artifact_uri,),
                    status=relationship_status,
                    obligation_uris=(claim.artifact_uri,),
                    verification_record_uri=record_uri if verified else None,
                ),
            ),
            obligations=(
                CapabilityObligation(
                    obligation_uri=claim.artifact_uri,
                    status=obligation_status,
                    verification_record_uri=record_uri if verified else None,
                ),
            ),
            assurance=CapabilityAssurance(
                level=assurance_level,
                basis=(
                    f"{verified_replay_basis}; external-domain completeness and "
                    "member/case semantics were not checked"
                    if verified
                    else "partition was proposed and inspected by its generator only"
                ),
                verification_record_uri=record_uri if verified else None,
            ),
            artifact_uris=tuple(artifact_uris),
        )

    def _verify(
        self,
        *,
        scope_uri: str,
        claim_uri: str,
        partition_uri: str,
    ) -> tuple[str, ResultEnvelope]:
        checker_id = self.installation.checker_id
        if checker_id is None:
            raise ValueError("finite partition checker is not authorized")
        scope = self.store.get(scope_uri)
        claim = self.store.get(claim_uri)
        partition = self.store.get(partition_uri)
        semantics = self.store.get(self.installation.semantics_uri)
        bindings = EvidenceBindings(
            claim_digest=claim.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=partition.manifest.object_digest,
            scope_digest=scope.manifest.object_digest,
        )
        payload: dict[str, Any] = {
            "replay": "equality-based finite coverage and conditional disjointness",
            "relation_id": "case.relation.partitions",
            "obligation_uri": claim_uri,
        }
        envelope = CertificateEnvelope(
            certificate_type="finite.partition",
            format_version="1",
            bindings=bindings,
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
            ),
            payload=payload,
        )
        certificate = self.artifacts.put(
            schema_uri=self.installation.certificate_schema_uri,
            semantics_uri=self.installation.semantics_uri,
            payload=envelope.model_dump(mode="json"),
            parents=(claim_uri, partition_uri, scope_uri),
            summary="finite partition replay certificate",
        )
        result = self.verification.verify_certificate(
            certificate_uri=certificate.artifact_uri,
            checker_id=checker_id,
        )
        return certificate.artifact_uri, result


def _partition_diagnostics(
    universe: list[str],
    cases: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    universe_set = set(universe)
    counts: dict[str, int] = {}
    for case in cases:
        for member in case["members"]:
            counts[member] = counts.get(member, 0) + 1
    missing = sorted(universe_set - set(counts))
    outside = sorted(set(counts) - universe_set)
    overlaps = sorted(member for member, count in counts.items() if count > 1)
    case_id_counts: dict[str, int] = {}
    for case in cases:
        case_id = case["case_id"]
        case_id_counts[case_id] = case_id_counts.get(case_id, 0) + 1
    duplicate_case_ids = sorted(
        case_id for case_id, count in case_id_counts.items() if count > 1
    )
    return missing, outside, overlaps, duplicate_case_ids
