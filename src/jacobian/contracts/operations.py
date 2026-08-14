"""Model-facing contracts for extensible mathematical operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic.json_schema import SkipJsonSchema

from jacobian.canonical import canonicalize_json
from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest, ValueUri
from jacobian.contracts.results import ContractModel, Execution, ExecutionStatus

OperationId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$",
        min_length=3,
        max_length=128,
        strict=True,
    ),
]


class OperationInputKind(StrEnum):
    """Coarse input boundary used to prevent incompatible discovery routes."""

    STRUCTURED_REQUEST = "STRUCTURED_REQUEST"
    FORMAL_PROPOSITION = "FORMAL_PROPOSITION"
    TYPED_ARTIFACT = "TYPED_ARTIFACT"


def _validate_descriptor_input_contract(
    accepted_input_kinds: tuple[OperationInputKind, ...],
    accepted_artifact_types: tuple[ArtifactUri, ...],
) -> None:
    if not accepted_input_kinds:
        raise ValueError("an operation must accept at least one input kind")
    if len(set(accepted_input_kinds)) != len(accepted_input_kinds):
        raise ValueError("accepted input kinds must be unique")
    if len(set(accepted_artifact_types)) != len(accepted_artifact_types):
        raise ValueError("accepted artifact types must be unique")
    accepts_typed_artifact = OperationInputKind.TYPED_ARTIFACT in accepted_input_kinds
    if accepted_artifact_types and not accepts_typed_artifact:
        raise ValueError("accepted artifact types require TYPED_ARTIFACT input")
    if accepts_typed_artifact and not accepted_artifact_types:
        raise ValueError("TYPED_ARTIFACT input requires accepted artifact types")


class OperationExample(ContractModel):
    """One operator-authored, schema-valid example."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        min_length=1,
        max_length=64,
    )
    description: str = Field(min_length=1, max_length=256)
    input: dict[str, Any]

    @model_validator(mode="after")
    def require_canonical_input(self) -> Self:
        canonicalize_json(self.input)
        return self


class OperationValuePort(ContractModel):
    """One named whole-value composition port."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value_type: str = Field(min_length=1, max_length=128)


class OperationDiscoveryRequest(ContractModel):
    """Compact installed-portfolio search, independent of any transport."""

    query: str = Field(min_length=1, max_length=512)
    domain: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$",
    )
    input_kind: OperationInputKind | None = None
    artifact_type: ArtifactUri | None = None
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    cursor: OperationId | None = None

    @model_validator(mode="after")
    def reject_blank_filters(self) -> Self:
        if not self.query.strip():
            raise ValueError("query must contain a non-whitespace character")
        if self.domain is not None and not self.domain.strip():
            raise ValueError("domain must contain a non-whitespace character")
        if self.artifact_type is not None and (
            self.input_kind is not OperationInputKind.TYPED_ARTIFACT
        ):
            raise ValueError("artifact_type requires input_kind=TYPED_ARTIFACT")
        if (
            self.input_kind is OperationInputKind.TYPED_ARTIFACT
            and self.artifact_type is None
        ):
            raise ValueError("TYPED_ARTIFACT input requires artifact_type")
        return self


class OperationDiscoveryMatch(ContractModel):
    """One compact installed outcome returned by operation discovery."""

    operation_id: OperationId
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    tags: tuple[str, ...] = ()
    relevance_score: int = Field(default=0, ge=0, strict=True)
    applicability: Literal[
        "INCOMPATIBLE",
        "NEEDS_MORE_TYPED_REQUIREMENTS",
    ]
    applicability_code: Literal[
        "FULL_REQUEST_REQUIRED",
        "INPUT_KIND_MISMATCH",
        "ARTIFACT_TYPE_MISMATCH",
    ]


class OperationDiscoveryResult(ContractModel):
    """Deterministically ranked compact installed outcomes."""

    discovery_version: Literal["1"] = "1"
    query: str
    domain: str | None = None
    input_kind: OperationInputKind | None = None
    artifact_type: ArtifactUri | None = None
    matches: tuple[OperationDiscoveryMatch, ...]
    total_matches: int = Field(ge=0, strict=True)
    truncated: bool
    next_cursor: OperationId | None = None

    @model_validator(mode="after")
    def bind_page_metadata(self) -> Self:
        operation_ids = tuple(match.operation_id for match in self.matches)
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("discovery matches must have unique operation IDs")
        if self.total_matches < len(self.matches):
            raise ValueError("total_matches cannot be smaller than the returned page")
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("truncated must agree with next_cursor")
        if self.next_cursor is not None and (
            not operation_ids or self.next_cursor != operation_ids[-1]
        ):
            raise ValueError("next_cursor must identify the final returned match")
        return self


class ProviderInstallTier(StrEnum):
    """Operational cost and isolation required to install one provider."""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class ProviderAvailability(StrEnum):
    """Whether this exact provider runtime is callable in the current process."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ProviderDigestKind(StrEnum):
    """What immutable provider material the runtime digest covers."""

    SOURCE_TREE = "SOURCE_TREE"
    PYTHON_DISTRIBUTION_RECORD = "PYTHON_DISTRIBUTION_RECORD"
    EXECUTABLE = "EXECUTABLE"
    COMPOSITE = "COMPOSITE"


def _validate_distribution_probe_attributes(
    distribution_required_attributes: tuple[str, ...],
) -> None:
    if len(set(distribution_required_attributes)) != len(
        distribution_required_attributes
    ):
        raise ValueError("provider feature probe attributes must be unique")
    if any(
        not attribute.isidentifier() for attribute in distribution_required_attributes
    ):
        raise ValueError("provider feature probe attributes must be identifiers")


def _validate_distribution_import_name(distribution_import_name: str | None) -> None:
    if distribution_import_name is not None and any(
        not component.isidentifier()
        for component in distribution_import_name.split(".")
    ):
        raise ValueError("provider distribution import name is invalid")


def _require_python_distribution_digest_for_probes(
    distribution_import_name: str | None,
    distribution_required_attributes: tuple[str, ...],
    digest_kind: ProviderDigestKind | None,
) -> None:
    if (distribution_import_name is not None or distribution_required_attributes) and (
        digest_kind is not ProviderDigestKind.PYTHON_DISTRIBUTION_RECORD
    ):
        raise ValueError(
            "provider distribution probes require a Python distribution digest"
        )


def _validate_provider_availability_identity(
    availability: ProviderAvailability,
    version: str | None,
    digest: Sha256Digest | None,
    digest_kind: ProviderDigestKind | None,
    diagnostic: str | None,
) -> None:
    if availability is ProviderAvailability.AVAILABLE:
        if version is None or digest is None or digest_kind is None:
            raise ValueError(
                "available provider runtime requires version, digest, and digest kind"
            )
        if diagnostic is not None:
            raise ValueError(
                "available provider runtime cannot carry an unavailable diagnostic"
            )
    elif diagnostic is None:
        raise ValueError("unavailable provider runtime requires a diagnostic")


def _validate_provider_collection_uniqueness(
    features: tuple[str, ...],
    checker_ids: tuple[CheckerUri, ...],
    license_files: tuple[str, ...],
) -> None:
    if len(set(features)) != len(features):
        raise ValueError("provider features must be unique")
    if len(set(checker_ids)) != len(checker_ids):
        raise ValueError("provider checker IDs must be unique")
    if len(set(license_files)) != len(license_files):
        raise ValueError("provider license files must be unique")


def _validate_provider_feature_labels(features: tuple[str, ...]) -> None:
    for feature in features:
        if not feature or len(feature) > 128:
            raise ValueError("provider features must contain 1 to 128 characters")


def _validate_provider_license_file_paths(license_files: tuple[str, ...]) -> None:
    for license_file in license_files:
        path = license_file.replace("\\", "/")
        if (
            not path
            or len(path) > 256
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("provider license files must be normalized relative paths")


class ProviderObservation(ContractModel):
    """Exact runtime identity and operator-facing availability metadata."""

    runtime_version: Literal["1"] = "1"
    provider: str = Field(
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$",
        min_length=3,
        max_length=128,
    )
    availability: ProviderAvailability
    version: str | None = Field(default=None, min_length=1, max_length=128)
    digest: Sha256Digest | None = None
    digest_kind: ProviderDigestKind | None = None
    platform: str = Field(min_length=1, max_length=128)
    install_tier: ProviderInstallTier
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
        _validate_distribution_probe_attributes(self.distribution_required_attributes)
        _validate_distribution_import_name(self.distribution_import_name)
        _require_python_distribution_digest_for_probes(
            self.distribution_import_name,
            self.distribution_required_attributes,
            self.digest_kind,
        )
        _validate_provider_availability_identity(
            self.availability,
            self.version,
            self.digest,
            self.digest_kind,
            self.diagnostic,
        )
        _validate_provider_collection_uniqueness(
            self.features,
            self.checker_ids,
            self.license_files,
        )
        _validate_provider_feature_labels(self.features)
        _validate_provider_license_file_paths(self.license_files)
        canonicalize_json(self.configuration)
        return self


class OperationDescriptor(ContractModel):
    """One installed operation advertised by an operator-installed adapter."""

    descriptor_version: Literal["1"] = "1"
    operation_id: OperationId
    version: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    provider: str = Field(min_length=1, max_length=128)
    # Execution/checker identity is internal lifecycle state, not catalog data.
    provider_runtime: SkipJsonSchema[ProviderObservation | None] = Field(
        default=None,
        exclude=True,
    )
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = False
    tags: tuple[str, ...] = ()
    accepted_input_kinds: tuple[OperationInputKind, ...] = (
        OperationInputKind.STRUCTURED_REQUEST,
    )
    accepted_artifact_types: tuple[ArtifactUri, ...] = ()
    produced_artifact_types: tuple[ArtifactUri, ...] = ()
    input_ports: tuple[OperationValuePort, ...] = ()
    output_ports: tuple[OperationValuePort, ...] = ()
    examples: tuple[OperationExample, ...] = ()

    @model_validator(mode="after")
    def require_canonical_schemas(self) -> Self:
        _validate_descriptor_input_contract(
            self.accepted_input_kinds,
            self.accepted_artifact_types,
        )
        if len(set(self.produced_artifact_types)) != len(self.produced_artifact_types):
            raise ValueError("produced artifact types must be unique")
        for ports, label in (
            (self.input_ports, "input"),
            (self.output_ports, "output"),
        ):
            names = tuple(port.name for port in ports)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} port names must be unique")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("operation invocation example names must be unique")
        canonicalize_json(self.input_schema)
        canonicalize_json(self.output_schema)
        if (
            self.provider_runtime is not None
            and self.provider_runtime.provider != self.provider
        ):
            raise ValueError("descriptor provider must match provider runtime identity")
        return self


class OperationRequest(ContractModel):
    request_version: Literal["1"] = "1"
    operation_id: OperationId
    input: dict[str, Any]
    inputs: dict[str, ValueUri] = Field(default_factory=dict)


class OperationDiagnostic(ContractModel):
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


def _validate_operation_execution_lane(
    execution_status: ExecutionStatus,
    diagnostics: tuple[OperationDiagnostic, ...],
    verification_record_uri: ArtifactUri | None,
) -> None:
    if execution_status is ExecutionStatus.COMPLETED and diagnostics:
        raise ValueError("completed operation execution cannot carry diagnostics")
    if (
        execution_status is not ExecutionStatus.COMPLETED
        and verification_record_uri is not None
    ):
        raise ValueError(
            "failed operation execution cannot carry a verification record"
        )


class OperationResult(ContractModel):
    """Operation invocation result."""

    response_version: Literal["2"] = "2"
    operation_id: OperationId
    operation_version: str = Field(min_length=1, max_length=64)
    execution: Execution
    output: dict[str, Any] = Field(default_factory=dict)
    diagnostics: tuple[OperationDiagnostic, ...] = ()
    verification_record_uri: ArtifactUri | None = None
    artifact_uris: tuple[ArtifactUri, ...] = ()

    @model_validator(mode="after")
    def enforce_lane_and_canonical_output(self) -> Self:
        canonicalize_json(self.output)
        _validate_operation_execution_lane(
            self.execution.status,
            self.diagnostics,
            self.verification_record_uri,
        )
        if (
            self.verification_record_uri is not None
            and self.verification_record_uri not in self.artifact_uris
        ):
            raise ValueError(
                "the verification record must be included in artifact_uris"
            )
        return self


class OperationCatalogSnapshot(ContractModel):
    catalog_version: Literal["1"] = "1"
    policy_profile: str = Field(min_length=1, max_length=64)
    policy_digest: Sha256Digest
    operations: tuple[OperationDescriptor, ...]

    @model_validator(mode="after")
    def require_unique_sorted_operations(self) -> Self:
        operation_ids = tuple(descriptor.operation_id for descriptor in self.operations)
        if operation_ids != tuple(sorted(set(operation_ids))):
            raise ValueError("catalog operation IDs must be unique and sorted")
        return self
