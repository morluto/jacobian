"""Finite case partition tools: produce a partition, optionally check it."""

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

_INPUT_SCHEMA: dict[str, Any] = {
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
}

_OUTPUT_SCHEMA: dict[str, Any] = {
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
}


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
) -> tuple[
    FinitePartitionAdapter,
    FinitePartitionVerifyAdapter | None,
    FinitePartitionInstallation,
]:
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
    installation = FinitePartitionInstallation(
        checker_id=registration.checker_id,
        semantics_uri=semantics_uri,
        scope_schema_uri=scope_schema_uri,
        claim_schema_uri=claim_schema_uri,
        partition_schema_uri=partition_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
    )
    producer = FinitePartitionAdapter(artifacts, store, installation)
    checker: FinitePartitionVerifyAdapter | None = None
    if installation.checker_id is not None:
        checker = FinitePartitionVerifyAdapter(
            artifacts, store, verification, installation
        )
    return producer, checker, installation


class FinitePartitionAdapter:
    """Materialize a finite partition (ordinary math tool)."""

    def __init__(
        self,
        artifacts: ArtifactService,
        store: ArtifactRepository,
        installation: FinitePartitionInstallation,
    ) -> None:
        self.artifacts = artifacts
        self.store = store
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="case.partition.finite",
            version="1",
            title="Partition an explicit finite domain",
            description=(
                "Materialize named cases over an explicit finite scope. "
                "Members and case labels are opaque caller-supplied strings. "
                "Independent coverage replay is the separate tool "
                "case.partition.finite.verify when a checker is authorized."
            ),
            provider="jacobian.finite",
            provider_runtime=known_provider_runtime(
                "jacobian.finite",
                features=("finite-partition",),
                checker_ids=(),
            ),
            input_schema=_INPUT_SCHEMA,
            output_schema=_OUTPUT_SCHEMA,
            tags=("cases", "finite", "coverage"),
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
        material = _materialize(self.artifacts, self.installation, request.input)
        return _partition_result(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            started=started,
            material=material,
            verification_result=None,
            certificate_uri=None,
        )


class FinitePartitionVerifyAdapter:
    """Independently check a finite partition (separate checker tool)."""

    def __init__(
        self,
        artifacts: ArtifactService,
        store: ArtifactRepository,
        verification: VerificationService,
        installation: FinitePartitionInstallation,
    ) -> None:
        if installation.checker_id is None:
            raise ValueError("finite partition checker is not authorized")
        self.artifacts = artifacts
        self.store = store
        self.verification = verification
        self.installation = installation
        self._descriptor = CapabilityDescriptor(
            capability_id="case.partition.finite.verify",
            version="1",
            title="Verify a finite partition",
            description=(
                "Replay equality-based coverage and optional disjointness for a "
                "caller-supplied finite partition with an authorized checker. "
                "Does not establish external-domain completeness or member semantics."
            ),
            provider="jacobian.finite",
            provider_runtime=known_provider_runtime(
                "jacobian.finite",
                features=("finite-partition",),
                checker_ids=(installation.checker_id,),
            ),
            input_schema=_INPUT_SCHEMA,
            output_schema=_OUTPUT_SCHEMA,
            tags=("cases", "finite", "coverage", "verification"),
            invocation_examples=(
                example(
                    "singleton_partition_check",
                    "Check a singleton partition covers its universe.",
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
        material = _materialize(self.artifacts, self.installation, request.input)
        certificate_uri, verification_result = _verify_partition(
            artifacts=self.artifacts,
            store=self.store,
            verification=self.verification,
            installation=self.installation,
            scope_uri=material.scope_uri,
            claim_uri=material.claim_uri,
            partition_uri=material.partition_uri,
        )
        return _partition_result(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            started=started,
            material=material,
            verification_result=verification_result,
            certificate_uri=certificate_uri,
        )


@dataclass(frozen=True, slots=True)
class _MaterializedPartition:
    scope_uri: str
    claim_uri: str
    partition_uri: str
    universe: list[str]
    require_disjoint: bool
    missing: list[str]
    outside: list[str]
    overlaps: list[str]
    duplicate_case_ids: list[str]


def _materialize(
    artifacts: ArtifactService,
    installation: FinitePartitionInstallation,
    payload: dict[str, Any],
) -> _MaterializedPartition:
    universe = [str(item) for item in payload["universe"]]
    cases = [
        {
            "case_id": str(case["case_id"]),
            "members": [str(member) for member in case["members"]],
        }
        for case in payload["cases"]
    ]
    require_disjoint = bool(payload.get("require_disjoint", True))
    scope = artifacts.put(
        schema_uri=installation.scope_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={"scope_schema_version": "1", "elements": universe},
        summary="explicit finite case scope",
    )
    claim = artifacts.put(
        schema_uri=installation.claim_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "predicate": "finite_partition",
            "require_disjoint": require_disjoint,
        },
        parents=(scope.artifact_uri,),
        summary="finite partition coverage obligation",
    )
    partition = artifacts.put(
        schema_uri=installation.partition_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload={"partition_schema_version": "1", "cases": cases},
        parents=(scope.artifact_uri, claim.artifact_uri),
        summary="proposed finite partition",
    )
    missing, outside, overlaps, duplicate_case_ids = _partition_diagnostics(
        universe, cases
    )
    return _MaterializedPartition(
        scope_uri=scope.artifact_uri,
        claim_uri=claim.artifact_uri,
        partition_uri=partition.artifact_uri,
        universe=universe,
        require_disjoint=require_disjoint,
        missing=missing,
        outside=outside,
        overlaps=overlaps,
        duplicate_case_ids=duplicate_case_ids,
    )


def _verify_partition(
    *,
    artifacts: ArtifactService,
    store: ArtifactRepository,
    verification: VerificationService,
    installation: FinitePartitionInstallation,
    scope_uri: str,
    claim_uri: str,
    partition_uri: str,
) -> tuple[str, ResultEnvelope]:
    checker_id = installation.checker_id
    if checker_id is None:
        raise ValueError("finite partition checker is not authorized")
    scope = store.get(scope_uri)
    claim = store.get(claim_uri)
    partition = store.get(partition_uri)
    semantics = store.get(installation.semantics_uri)
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
    certificate = artifacts.put(
        schema_uri=installation.certificate_schema_uri,
        semantics_uri=installation.semantics_uri,
        payload=envelope.model_dump(mode="json"),
        parents=(claim_uri, partition_uri, scope_uri),
        summary="finite partition replay certificate",
    )
    result = verification.verify_certificate(
        certificate_uri=certificate.artifact_uri,
        checker_id=checker_id,
    )
    return certificate.artifact_uri, result


def _partition_result(
    *,
    capability_id: str,
    capability_version: str,
    started: float,
    material: _MaterializedPartition,
    verification_result: ResultEnvelope | None,
    certificate_uri: str | None,
) -> CapabilityResult:
    record_uri = (
        verification_result.verification_record_uri
        if verification_result is not None
        else None
    )
    verified = (
        verification_result is not None
        and verification_result.assurance.verification is Verification.VERIFIED
    )
    artifact_uris = [
        material.scope_uri,
        material.claim_uri,
        material.partition_uri,
    ]
    if certificate_uri is not None:
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
        and not material.missing
        and not material.outside
        and not material.duplicate_case_ids
        and (not material.require_disjoint or not material.overlaps)
    )
    verified_replay_basis = (
        "authorized checker replayed equality-based coverage and required "
        "disjointness within the caller-supplied universe"
        if material.require_disjoint
        else (
            "authorized checker replayed equality-based coverage within the "
            "caller-supplied universe; disjointness was not required"
        )
    )
    return CapabilityResult(
        capability_id=capability_id,
        capability_version=capability_version,
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
            "scope_uri": material.scope_uri,
            "claim_uri": material.claim_uri,
            "partition_uri": material.partition_uri,
            "certificate_uri": certificate_uri,
            "verification_record_uri": record_uri,
            "missing": material.missing,
            "outside": material.outside,
            "overlaps": material.overlaps,
            "duplicate_case_ids": material.duplicate_case_ids,
        },
        scope=CapabilityScope(
            description=(
                "the exact caller-supplied finite universe; external-domain "
                "completeness and member semantics are not checked"
            ),
            parameters={"element_count": len(material.universe)},
            artifact_uri=material.scope_uri,
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
                source_artifact_uris=(material.scope_uri,),
                target_artifact_uris=(material.partition_uri,),
                status=relationship_status,
                obligation_uris=(material.claim_uri,),
                verification_record_uri=record_uri if verified else None,
            ),
        ),
        obligations=(
            CapabilityObligation(
                obligation_uri=material.claim_uri,
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
