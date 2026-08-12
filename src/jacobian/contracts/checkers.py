"""Operator-managed checker registry contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts._verification_rules import (
    validate_certified_relationship_endpoints,
    validate_decisive_replayable_evidence,
)
from jacobian.contracts.capabilities import (
    CapabilityId,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.evidence import FormatIdentifier
from jacobian.contracts.results import (
    Arithmetic,
    Conclusion,
    ContractModel,
    Coverage,
    Method,
)

Entrypoint = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
        strict=True,
    ),
]

CheckerModuleName = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$",
        strict=True,
    ),
]


class EvidenceKind(StrEnum):
    WITNESS = "WITNESS"
    CERTIFICATE = "CERTIFICATE"


class CheckerSourceModule(ContractModel):
    """One measured first-party Python module used by a checker worker."""

    module: CheckerModuleName
    source_digest: Sha256Digest


class CheckerPythonRuntime(ContractModel):
    """Exact interpreter identity for an isolated checker process."""

    implementation: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=128)
    executable_digest: Sha256Digest


class CheckerPythonDistribution(ContractModel):
    """One Python distribution bound to its complete installed file closure."""

    distribution: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    file_count: int = Field(ge=1, strict=True)
    files_digest: Sha256Digest


class CheckerSandboxPolicy(ContractModel):
    """The concrete limits and import guard used for one checker process."""

    policy_version: Literal["1"] = "1"
    process_policy: Literal["jacobian.bounded-checker-worker/v1"] = (
        "jacobian.bounded-checker-worker/v1"
    )
    first_party_import_policy: Literal["declared-source-only/v1"] = (
        "declared-source-only/v1"
    )
    working_directory_policy: Literal["inherited-server-cwd"] = "inherited-server-cwd"
    network_policy: Literal["host-network-unrestricted"] = "host-network-unrestricted"
    max_wall_seconds: int = Field(ge=1, le=300, strict=True)
    max_cpu_seconds: int = Field(ge=1, le=301, strict=True)
    max_address_space_bytes: int = Field(ge=1, strict=True)
    max_stdout_bytes: int = Field(ge=1, strict=True)
    max_stderr_bytes: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def require_cpu_headroom(self) -> Self:
        if self.max_cpu_seconds < self.max_wall_seconds:
            raise ValueError("checker CPU limit cannot be smaller than wall limit")
        return self


class CheckerManifest(ContractModel):
    """Versioned, remeasurable execution identity for one checker."""

    manifest_schema_version: Literal["2"] = "2"
    entrypoint: Entrypoint
    checker_source_modules: tuple[CheckerSourceModule, ...]
    worker_source_modules: tuple[CheckerSourceModule, ...]
    python_runtime: CheckerPythonRuntime
    python_distributions: tuple[CheckerPythonDistribution, ...]
    provider_runtime: CapabilityProviderRuntime | None = None
    passive_contract_uris: tuple[ArtifactUri, ...]
    sandbox: CheckerSandboxPolicy

    @model_validator(mode="after")
    def require_closed_identity(self) -> Self:
        _require_sorted_source_modules(
            self.checker_source_modules,
            scope="checker",
        )
        _require_sorted_source_modules(
            self.worker_source_modules,
            scope="worker",
        )
        entrypoint_module = self.entrypoint.partition(":")[0]
        if entrypoint_module not in {
            source.module for source in self.checker_source_modules
        }:
            raise ValueError("checker manifest must bind its entrypoint module")
        if "jacobian.checker_worker" not in {
            source.module for source in self.worker_source_modules
        }:
            raise ValueError("checker manifest must bind the checker worker runtime")
        _require_consistent_source_groups(
            self.checker_source_modules,
            self.worker_source_modules,
        )
        _require_python_distributions(self.python_distributions)
        if self.passive_contract_uris != tuple(
            sorted(self.passive_contract_uris)
        ) or len(set(self.passive_contract_uris)) != len(self.passive_contract_uris):
            raise ValueError(
                "checker manifest passive contracts must be unique and sorted"
            )
        _require_exact_external_runtime(self.provider_runtime, self.entrypoint)
        return self


_REQUIRED_WORKER_DISTRIBUTIONS = frozenset({"pydantic", "pydantic-core", "rfc8785"})


def _require_sorted_source_modules(
    modules: tuple[CheckerSourceModule, ...],
    *,
    scope: str,
) -> None:
    names = tuple(source.module for source in modules)
    if not names:
        raise ValueError(f"checker manifest must bind {scope} source modules")
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(
            f"checker manifest {scope} source modules must be unique and sorted"
        )


def _require_consistent_source_groups(
    checker_modules: tuple[CheckerSourceModule, ...],
    worker_modules: tuple[CheckerSourceModule, ...],
) -> None:
    measured = {source.module: source.source_digest for source in checker_modules}
    for source in worker_modules:
        digest = measured.setdefault(source.module, source.source_digest)
        if digest != source.source_digest:
            raise ValueError(
                "checker and worker source groups disagree about a module digest"
            )


def _require_python_distributions(
    distributions: tuple[CheckerPythonDistribution, ...],
) -> None:
    names = tuple(_distribution_key(item.distribution) for item in distributions)
    if not names:
        raise ValueError("checker manifest must bind Python distributions")
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(
            "checker manifest Python distributions must be unique and sorted"
        )
    missing = _REQUIRED_WORKER_DISTRIBUTIONS - set(names)
    if missing:
        raise ValueError(
            "checker manifest must bind worker Python distributions: "
            + ", ".join(sorted(missing))
        )


def _distribution_key(name: str) -> str:
    """Return the PEP 503 comparison key without adding a packaging dependency."""

    return re.sub(r"[-_.]+", "-", name).lower()


class CheckerRegistration(ContractModel):
    checker_schema_version: Literal["3"] = "3"
    checker_id: CheckerUri
    name: str
    implementation: CheckerManifest
    implementation_digest: Sha256Digest
    evidence_kind: EvidenceKind
    format_id: FormatIdentifier
    format_version: str
    claim_schema_uris: tuple[ArtifactUri, ...] = ()
    semantics_uris: tuple[ArtifactUri, ...] = ()
    candidate_schema_uris: tuple[ArtifactUri, ...] = ()
    target_schema_uris: tuple[ArtifactUri, ...] = ()
    target_semantics_uris: tuple[ArtifactUri, ...] = ()
    authorized: bool = True

    @model_validator(mode="after")
    def require_manifest_scope(self) -> Self:
        expected_contracts = tuple(
            sorted(
                {
                    *self.claim_schema_uris,
                    *self.semantics_uris,
                    *self.candidate_schema_uris,
                    *self.target_schema_uris,
                    *self.target_semantics_uris,
                }
            )
        )
        if self.implementation.passive_contract_uris != expected_contracts:
            raise ValueError(
                "checker manifest passive contracts must match registration scope"
            )
        return self


def _require_exact_external_runtime(
    runtime: CapabilityProviderRuntime | None,
    entrypoint: str,
) -> None:
    """Reject a provider identity that cannot be remeasured by the worker."""

    if runtime is None:
        return
    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.digest_kind
        not in {
            CapabilityProviderDigestKind.EXECUTABLE,
            CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
            CapabilityProviderDigestKind.SOURCE_TREE,
            CapabilityProviderDigestKind.COMPOSITE,
        }
        or runtime.digest is None
    ):
        raise ValueError(
            "checker provider runtime must identify an available executable, "
            "Python distribution, remeasurable source tree, or fully bound composite"
        )
    if (
        runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE
        and not isinstance(runtime.configuration.get("executable"), str)
    ):
        raise ValueError("checker executable runtime must name its executable")
    if runtime.digest_kind is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD:
        distribution = runtime.configuration.get("distribution")
        import_name = runtime.configuration.get("import_name")
        if not isinstance(distribution, str) or not isinstance(import_name, str):
            raise ValueError(
                "checker Python distribution runtime must name its distribution and import"
            )
    if runtime.digest_kind is CapabilityProviderDigestKind.SOURCE_TREE:
        runtime_entrypoint = runtime.configuration.get("entrypoint")
        if not isinstance(runtime_entrypoint, str):
            raise ValueError("checker source runtime must name its entrypoint")
        if runtime_entrypoint != entrypoint:
            raise ValueError(
                "checker source runtime entrypoint must bind the checker entrypoint"
            )
    if runtime.checker_ids:
        raise ValueError(
            "checker provider runtime cannot recursively contain checker IDs"
        )


class CheckerAuditEvent(ContractModel):
    sequence: int
    checker_id: CheckerUri
    action: Literal["AUTHORIZED", "REVOKED"]
    reason: str
    recorded_at: str


def _validate_rejected_checker_evidence(
    accepted: bool,
    conclusion: Conclusion,
    relation_id: CapabilityId | None,
    relationship_source_artifact_uris: tuple[ArtifactUri, ...],
    relationship_target_artifact_uris: tuple[ArtifactUri, ...],
    obligation_uri: ArtifactUri | None,
) -> None:
    if not accepted and conclusion not in {
        Conclusion.UNKNOWN,
        Conclusion.NOT_APPLICABLE,
    }:
        raise ValueError("a rejected checker input cannot decide the claim")
    if not accepted and (
        relation_id is not None
        or relationship_source_artifact_uris
        or relationship_target_artifact_uris
        or obligation_uri is not None
    ):
        raise ValueError("rejected evidence cannot certify relationship metadata")


class CheckerDecision(ContractModel):
    accepted: bool
    conclusion: Conclusion
    arithmetic: Arithmetic
    method: Method
    coverage: Coverage
    detail: str = ""
    relation_id: CapabilityId | None = None
    relationship_source_artifact_uris: tuple[ArtifactUri, ...] = ()
    relationship_target_artifact_uris: tuple[ArtifactUri, ...] = ()
    obligation_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def rejected_evidence_has_no_mathematical_conclusion(self) -> Self:
        _validate_rejected_checker_evidence(
            self.accepted,
            self.conclusion,
            self.relation_id,
            self.relationship_source_artifact_uris,
            self.relationship_target_artifact_uris,
            self.obligation_uri,
        )
        validate_certified_relationship_endpoints(
            self.relation_id,
            self.relationship_source_artifact_uris,
            self.relationship_target_artifact_uris,
        )
        if self.accepted:
            validate_decisive_replayable_evidence(
                self.conclusion,
                self.arithmetic,
                self.coverage,
                self.method,
            )
        return self
