"""Model-facing contracts for extensible mathematical capabilities."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.results import ContractModel, Execution

CapabilityId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        min_length=3,
        max_length=128,
        strict=True,
    ),
]


class CapabilityMode(StrEnum):
    """The low-friction exploration and explicit verification lanes."""

    EXPLORE = "EXPLORE"
    VERIFY = "VERIFY"


class CapabilityInputKind(StrEnum):
    """Coarse input boundary used to prevent incompatible discovery routes."""

    STRUCTURED_REQUEST = "STRUCTURED_REQUEST"
    FORMAL_PROPOSITION = "FORMAL_PROPOSITION"
    TYPED_ARTIFACT = "TYPED_ARTIFACT"
    NATURAL_LANGUAGE_PROOF = "NATURAL_LANGUAGE_PROOF"


def _validate_descriptor_input_contract(
    accepted_input_kinds: tuple[CapabilityInputKind, ...],
    accepted_artifact_types: tuple[ArtifactUri, ...],
) -> None:
    if not accepted_input_kinds:
        raise ValueError("a capability must accept at least one input kind")
    if len(set(accepted_input_kinds)) != len(accepted_input_kinds):
        raise ValueError("accepted input kinds must be unique")
    if len(set(accepted_artifact_types)) != len(accepted_artifact_types):
        raise ValueError("accepted artifact types must be unique")
    accepts_typed_artifact = CapabilityInputKind.TYPED_ARTIFACT in accepted_input_kinds
    if accepted_artifact_types and not accepts_typed_artifact:
        raise ValueError("accepted artifact types require TYPED_ARTIFACT input")
    if accepts_typed_artifact and not accepted_artifact_types:
        raise ValueError("TYPED_ARTIFACT input requires accepted artifact types")


class CapabilityInvocationExample(ContractModel):
    """One operator-authored, schema-valid example for an advertised mode."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        min_length=1,
        max_length=64,
    )
    description: str = Field(min_length=1, max_length=256)
    mode: CapabilityMode
    input: dict[str, Any]

    @model_validator(mode="after")
    def require_canonical_input(self) -> Self:
        canonicalize_json(self.input)
        return self


class CapabilityDiscoveryRequest(ContractModel):
    """Compact installed-portfolio search, independent of any transport."""

    query: str | None = Field(default=None, min_length=1, max_length=512)
    domain: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    mode: CapabilityMode | None = None
    input_kind: CapabilityInputKind | None = None
    artifact_type: ArtifactUri | None = None
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    cursor: CapabilityId | None = None

    @model_validator(mode="after")
    def reject_blank_filters(self) -> Self:
        if self.query is not None and not self.query.strip():
            raise ValueError("query must contain a non-whitespace character")
        if self.domain is not None and not self.domain.strip():
            raise ValueError("domain must contain a non-whitespace character")
        if self.artifact_type is not None and (
            self.input_kind is not CapabilityInputKind.TYPED_ARTIFACT
        ):
            raise ValueError("artifact_type requires input_kind=TYPED_ARTIFACT")
        if (
            self.input_kind is CapabilityInputKind.TYPED_ARTIFACT
            and self.artifact_type is None
        ):
            raise ValueError("TYPED_ARTIFACT input requires artifact_type")
        return self


class CapabilityDiscoveryReformulateQueryRecoveryPath(ContractModel):
    """Offer a differently worded query without prescribing one."""

    action: Literal["reformulate_query"]
    tool: Literal["math.find"] = "math.find"
    change: Literal["Use different or broader mathematical language for query."] = (
        "Use different or broader mathematical language for query."
    )


class CapabilityDiscoveryRemoveUnknownDomainRecoveryPath(ContractModel):
    """Expose the rejected domain filter as one removable constraint."""

    action: Literal["remove_unknown_domain_filter"]
    tool: Literal["math.find"] = "math.find"
    rejected_domain: str = Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    change: Literal["Retry without the unrecognized domain filter."] = (
        "Retry without the unrecognized domain filter."
    )


class CapabilityDiscoveryRemoveFiltersRecoveryPath(ContractModel):
    """Offer unfiltered discovery without ranking it above other choices."""

    action: Literal["remove_filters"]
    tool: Literal["math.find"] = "math.find"
    change: Literal["Remove domain, mode, input_kind, or artifact_type filters."] = (
        "Remove domain, mode, input_kind, or artifact_type filters."
    )


class CapabilityDiscoveryBrowseRecoveryPath(ContractModel):
    """Expose the existing empty-query browse operation."""

    action: Literal["browse"]
    tool: Literal["math.find"] = "math.find"
    arguments: dict[str, Any] = Field(default_factory=dict, max_length=0)


class CapabilityDiscoveryInspectCatalogRecoveryPath(ContractModel):
    """Expose the complete catalog resource as an alternative access path."""

    action: Literal["inspect_catalog"]
    resource_uri: Literal["capability://catalog"] = "capability://catalog"


CapabilityDiscoveryRecoveryPath = Annotated[
    CapabilityDiscoveryReformulateQueryRecoveryPath
    | CapabilityDiscoveryRemoveUnknownDomainRecoveryPath
    | CapabilityDiscoveryRemoveFiltersRecoveryPath
    | CapabilityDiscoveryBrowseRecoveryPath
    | CapabilityDiscoveryInspectCatalogRecoveryPath,
    Field(discriminator="action"),
]


class CapabilityDiscoveryMatch(ContractModel):
    """One compact installed outcome returned by capability discovery."""

    capability_id: CapabilityId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    modes: tuple[CapabilityMode, ...]
    tags: tuple[str, ...] = ()
    matched_on: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    has_invocation_examples: bool = False
    relevance_score: int = Field(default=0, ge=0, strict=True)
    query_term_count: int = Field(default=0, ge=0, strict=True)
    query_coverage_milli: int = Field(default=0, ge=0, le=1000, strict=True)
    lexical_fit: Literal["STRONG_CANDIDATE", "WEAK_LEXICAL_MATCH"] = (
        "WEAK_LEXICAL_MATCH"
    )


class CapabilityDiscoveryResult(ContractModel):
    """Deterministically ranked compact installed outcomes."""

    discovery_version: Literal["1"] = "1"
    query: str | None = None
    domain: str | None = None
    domain_filter_status: Literal["UNFILTERED", "MATCHED", "UNKNOWN"] = "UNFILTERED"
    domain_filter_basis: str = Field(
        default="No domain filter was supplied.",
        min_length=1,
        max_length=512,
    )
    mode: CapabilityMode | None = None
    resolved_input_kind: CapabilityInputKind | None = None
    artifact_type: ArtifactUri | None = None
    routing_status: Literal["UNFILTERED", "ROUTES_FOUND", "NO_ROUTE"] = "UNFILTERED"
    routing_basis: str = Field(min_length=1, max_length=512)
    matches: tuple[CapabilityDiscoveryMatch, ...]
    total_matches: int = Field(ge=0, strict=True)
    truncated: bool
    next_cursor: CapabilityId | None = None
    available_domains: tuple[str, ...] = ()
    portfolio_fit: Literal[
        "UNFILTERED",
        "STRONG_CANDIDATES_FOUND",
        "ONLY_WEAK_LEXICAL_MATCHES",
        "NO_LEXICAL_MATCHES",
    ] = "UNFILTERED"
    portfolio_fit_basis: str = Field(min_length=1, max_length=512)


class CapabilityInstallTier(StrEnum):
    """Operational cost and isolation required to install one provider."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class CapabilityProviderAvailability(StrEnum):
    """Whether this exact provider runtime is callable in the current process."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityProviderDigestKind(StrEnum):
    """What immutable provider material the runtime digest covers."""

    SOURCE_TREE = "SOURCE_TREE"
    PYTHON_DISTRIBUTION_RECORD = "PYTHON_DISTRIBUTION_RECORD"
    EXECUTABLE = "EXECUTABLE"
    COMPOSITE = "COMPOSITE"


class CapabilityProviderRuntime(ContractModel):
    """Exact runtime identity and operator-facing availability metadata."""

    runtime_version: Literal["1"] = "1"
    provider: str = Field(
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
        min_length=3,
        max_length=128,
    )
    availability: CapabilityProviderAvailability
    version: str | None = Field(default=None, min_length=1, max_length=128)
    digest: Sha256Digest | None = None
    digest_kind: CapabilityProviderDigestKind | None = None
    platform: str = Field(min_length=1, max_length=128)
    install_tier: CapabilityInstallTier
    license_id: str = Field(min_length=1, max_length=128)
    license_files: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    checker_ids: tuple[CheckerUri, ...] = ()
    configuration: dict[str, Any] = Field(default_factory=dict)
    distribution_import_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    distribution_required_attributes: tuple[str, ...] = Field(
        default=(),
        max_length=64,
    )
    diagnostic: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> Self:
        if len(set(self.distribution_required_attributes)) != len(
            self.distribution_required_attributes
        ):
            raise ValueError("provider feature probe attributes must be unique")
        if any(
            not attribute.isidentifier()
            for attribute in self.distribution_required_attributes
        ):
            raise ValueError("provider feature probe attributes must be identifiers")
        if self.distribution_import_name is not None and any(
            not component.isidentifier()
            for component in self.distribution_import_name.split(".")
        ):
            raise ValueError("provider distribution import name is invalid")
        if (
            self.distribution_import_name is not None
            or self.distribution_required_attributes
        ) and (
            self.digest_kind
            is not CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
        ):
            raise ValueError(
                "provider distribution probes require a Python distribution digest"
            )
        if self.availability is CapabilityProviderAvailability.AVAILABLE:
            if self.version is None or self.digest is None or self.digest_kind is None:
                raise ValueError(
                    "available provider runtime requires version, digest, "
                    "and digest kind"
                )
            if self.diagnostic is not None:
                raise ValueError(
                    "available provider runtime cannot carry an unavailable diagnostic"
                )
        elif self.diagnostic is None:
            raise ValueError("unavailable provider runtime requires a diagnostic")
        if len(set(self.features)) != len(self.features):
            raise ValueError("provider features must be unique")
        if len(set(self.checker_ids)) != len(self.checker_ids):
            raise ValueError("provider checker IDs must be unique")
        if len(set(self.license_files)) != len(self.license_files):
            raise ValueError("provider license files must be unique")
        for feature in self.features:
            if not feature or len(feature) > 128:
                raise ValueError("provider features must contain 1 to 128 characters")
        for license_file in self.license_files:
            path = license_file.replace("\\", "/")
            if (
                not path
                or len(path) > 256
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                raise ValueError(
                    "provider license files must be normalized relative paths"
                )
        canonicalize_json(self.configuration)
        return self


class CapabilityAssuranceLevel(StrEnum):
    """Coarse model-facing assurance without hiding the detailed result record."""

    HEURISTIC = "HEURISTIC"
    COMPUTED = "COMPUTED"
    VERIFIED = "VERIFIED"


class CapabilityRelationshipStatus(StrEnum):
    """Whether a returned mathematical relationship has checker backing."""

    PROPOSED = "PROPOSED"
    VERIFIED = "VERIFIED"


class CapabilityObligationStatus(StrEnum):
    """Lifecycle of a proof obligation created by a capability."""

    OPEN = "OPEN"
    DISCHARGED = "DISCHARGED"


class CapabilityCompletenessStatus(StrEnum):
    """How much of the explicitly declared scope an operation covered."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


class CapabilityDescriptor(ContractModel):
    """One installed operation advertised by an operator-installed adapter."""

    descriptor_version: Literal["1"] = "1"
    capability_id: CapabilityId
    version: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    provider: str = Field(min_length=1, max_length=128)
    provider_runtime: CapabilityProviderRuntime | None = None
    modes: tuple[CapabilityMode, ...]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = False
    tags: tuple[str, ...] = ()
    accepted_input_kinds: tuple[CapabilityInputKind, ...] = (
        CapabilityInputKind.STRUCTURED_REQUEST,
    )
    accepted_artifact_types: tuple[ArtifactUri, ...] = ()
    produced_artifact_types: tuple[ArtifactUri, ...] = ()
    discovery_visible: bool = True
    invocation_examples: tuple[CapabilityInvocationExample, ...] = ()

    @model_validator(mode="after")
    def require_modes_and_canonical_schemas(self) -> Self:
        if not self.modes:
            raise ValueError("a capability must support at least one mode")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("capability modes must be unique")
        _validate_descriptor_input_contract(
            self.accepted_input_kinds,
            self.accepted_artifact_types,
        )
        if len(set(self.produced_artifact_types)) != len(self.produced_artifact_types):
            raise ValueError("produced artifact types must be unique")
        if len({example.name for example in self.invocation_examples}) != len(
            self.invocation_examples
        ):
            raise ValueError("capability invocation example names must be unique")
        unsupported_examples = [
            example.name
            for example in self.invocation_examples
            if example.mode not in self.modes
        ]
        if unsupported_examples:
            raise ValueError(
                "capability invocation examples must use advertised modes: "
                + ", ".join(unsupported_examples)
            )
        canonicalize_json(self.input_schema)
        canonicalize_json(self.output_schema)
        if (
            self.provider_runtime is not None
            and self.provider_runtime.provider != self.provider
        ):
            raise ValueError("descriptor provider must match provider runtime identity")
        return self


class CapabilityRequest(ContractModel):
    request_version: Literal["1"] = "1"
    capability_id: CapabilityId
    mode: CapabilityMode = CapabilityMode.EXPLORE
    input: dict[str, Any]


class CapabilityAssurance(ContractModel):
    level: CapabilityAssuranceLevel
    basis: str = Field(min_length=1, max_length=1024)
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def bind_verified_assurance(self) -> Self:
        if (
            self.level is CapabilityAssuranceLevel.VERIFIED
            and self.verification_record_uri is None
        ):
            raise ValueError("verified capability assurance requires a record URI")
        if (
            self.level is not CapabilityAssuranceLevel.VERIFIED
            and self.verification_record_uri is not None
        ):
            raise ValueError(
                "only verified capability assurance may carry a record URI"
            )
        return self


class CapabilityScope(ContractModel):
    """Domain-owned scope parameters, optionally materialized as an artifact."""

    description: str | None = Field(default=None, min_length=1, max_length=512)
    parameters: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_explicit_scope(self) -> Self:
        canonicalize_json(self.parameters)
        if not self.parameters and self.artifact_uri is None:
            raise ValueError("scope requires parameters or an artifact URI")
        return self


class CapabilityCompleteness(ContractModel):
    """Coverage claim over the result's exact declared scope."""

    status: CapabilityCompletenessStatus = CapabilityCompletenessStatus.NOT_APPLICABLE
    basis: str = Field(min_length=1, max_length=1024)
    assurance_level: CapabilityAssuranceLevel = CapabilityAssuranceLevel.HEURISTIC
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def bind_verified_completeness(self) -> Self:
        if (
            self.assurance_level is CapabilityAssuranceLevel.VERIFIED
            and self.verification_record_uri is None
        ):
            raise ValueError("verified completeness requires a record URI")
        if (
            self.assurance_level is not CapabilityAssuranceLevel.VERIFIED
            and self.verification_record_uri is not None
        ):
            raise ValueError("only verified completeness may carry a record URI")
        if (
            self.status is not CapabilityCompletenessStatus.COMPLETE
            and self.assurance_level is CapabilityAssuranceLevel.VERIFIED
        ):
            raise ValueError("only complete coverage may be independently verified")
        return self


class CapabilityRelationship(ContractModel):
    """A domain-owned relationship between exact immutable artifacts."""

    relation_id: CapabilityId
    source_artifact_uris: tuple[ArtifactUri, ...]
    target_artifact_uris: tuple[ArtifactUri, ...]
    status: CapabilityRelationshipStatus = CapabilityRelationshipStatus.PROPOSED
    obligation_uris: tuple[ArtifactUri, ...] = ()
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_bound_endpoints(self) -> Self:
        if not self.source_artifact_uris or not self.target_artifact_uris:
            raise ValueError("relationship requires source and target artifacts")
        if len(set(self.source_artifact_uris)) != len(self.source_artifact_uris):
            raise ValueError("relationship source artifacts must be unique")
        if len(set(self.target_artifact_uris)) != len(self.target_artifact_uris):
            raise ValueError("relationship target artifacts must be unique")
        if (
            self.status is CapabilityRelationshipStatus.VERIFIED
            and self.verification_record_uri is None
        ):
            raise ValueError("verified relationship requires a record URI")
        if (
            self.status is CapabilityRelationshipStatus.PROPOSED
            and self.verification_record_uri is not None
        ):
            raise ValueError("proposed relationship cannot carry a record URI")
        return self


class CapabilityObligation(ContractModel):
    """One materialized proof obligation and its checker-backed lifecycle."""

    obligation_uri: ArtifactUri
    status: CapabilityObligationStatus = CapabilityObligationStatus.OPEN
    verification_record_uri: ArtifactUri | None = None

    @model_validator(mode="after")
    def require_record_for_discharge(self) -> Self:
        if (
            self.status is CapabilityObligationStatus.DISCHARGED
            and self.verification_record_uri is None
        ):
            raise ValueError("discharged obligation requires a record URI")
        if (
            self.status is CapabilityObligationStatus.OPEN
            and self.verification_record_uri is not None
        ):
            raise ValueError("open obligation cannot carry a record URI")
        return self


class CapabilityDiagnostic(ContractModel):
    """Actionable, stage-aware failure information without a truth claim."""

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,63}$")
    stage: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    message: str = Field(min_length=1, max_length=1024)
    path: str | None = Field(default=None, max_length=512)
    schema_uri: ArtifactUri | None = None
    expected: str | None = Field(default=None, max_length=1024)
    actual_type: str | None = Field(default=None, max_length=128)
    hint: str | None = Field(default=None, max_length=1024)
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilityResult(ContractModel):
    """Capability invocation result."""

    response_version: Literal["2"] = "2"
    capability_id: CapabilityId
    capability_version: str = Field(min_length=1, max_length=64)
    mode: CapabilityMode
    execution: Execution
    output: dict[str, Any] = Field(default_factory=dict)
    scope: CapabilityScope | None = None
    completeness: CapabilityCompleteness = Field(
        default_factory=lambda: CapabilityCompleteness(
            basis="the operation makes no completeness claim",
        )
    )
    relationships: tuple[CapabilityRelationship, ...] = ()
    obligations: tuple[CapabilityObligation, ...] = ()
    diagnostics: tuple[CapabilityDiagnostic, ...] = ()
    assurance: CapabilityAssurance
    artifact_uris: tuple[ArtifactUri, ...] = ()
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    provider_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def enforce_lane_and_canonical_output(self) -> Self:
        canonicalize_json(self.output)
        if self.execution.status.value == "COMPLETED" and self.diagnostics:
            raise ValueError("completed capability execution cannot carry diagnostics")
        if (
            self.assurance.level is CapabilityAssuranceLevel.VERIFIED
            and self.mode is not CapabilityMode.VERIFY
        ):
            raise ValueError("the exploration lane cannot return verified assurance")
        if (
            self.execution.status.value != "COMPLETED"
            and self.assurance.level is CapabilityAssuranceLevel.VERIFIED
        ):
            raise ValueError("failed capability execution cannot be verified")
        if (
            self.completeness.status is CapabilityCompletenessStatus.COMPLETE
            and self.scope is None
        ):
            raise ValueError("complete result requires explicit scope")
        if (
            self.execution.status.value != "COMPLETED"
            and self.completeness.status is CapabilityCompletenessStatus.COMPLETE
        ):
            raise ValueError("failed execution cannot be complete")
        record_uri = self.assurance.verification_record_uri
        for relationship in self.relationships:
            if relationship.status is CapabilityRelationshipStatus.VERIFIED:
                if self.assurance.level is not CapabilityAssuranceLevel.VERIFIED:
                    raise ValueError(
                        "verified relationship requires verified result assurance"
                    )
                if relationship.verification_record_uri != record_uri:
                    raise ValueError(
                        "verified relationship must use the result verification record"
                    )
        for obligation in self.obligations:
            if obligation.status is CapabilityObligationStatus.DISCHARGED:
                if self.assurance.level is not CapabilityAssuranceLevel.VERIFIED:
                    raise ValueError(
                        "discharged obligation requires verified result assurance"
                    )
                if obligation.verification_record_uri != record_uri:
                    raise ValueError(
                        "discharged obligation must use the result verification record"
                    )
        if self.completeness.assurance_level is CapabilityAssuranceLevel.VERIFIED:
            if self.assurance.level is not CapabilityAssuranceLevel.VERIFIED:
                raise ValueError(
                    "verified completeness requires verified result assurance"
                )
            if self.completeness.verification_record_uri != record_uri:
                raise ValueError(
                    "verified completeness must use the result verification record"
                )
        return self


class CapabilityCatalog(ContractModel):
    catalog_version: Literal["1"] = "1"
    policy_profile: str = Field(min_length=1, max_length=64)
    policy_digest: Sha256Digest
    capabilities: tuple[CapabilityDescriptor, ...]
