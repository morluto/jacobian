"""Typed operations and bundles owned by mathematical domains."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from jacobian.checker_operations import ExactReplayCheckerDeclaration
    from jacobian.installation.context import InstallationContext
    from jacobian.operation_installation import InstalledDomainBundle

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus


@dataclass(frozen=True, slots=True)
class ComputedSuccess[ResultT: ContractModel]:
    """One complete, contract-valid result candidate."""

    value: ResultT


@dataclass(frozen=True, slots=True)
class ComputedNotApplicable:
    """Valid request outside an operation's mathematical domain."""

    diagnostic: CapabilityDiagnostic


@dataclass(frozen=True, slots=True)
class OperationExecutionFailure:
    """Operational failure that carries no mathematical conclusion."""

    status: ExecutionStatus
    diagnostic: CapabilityDiagnostic

    def __post_init__(self) -> None:
        if self.status not in {
            ExecutionStatus.ERROR,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }:
            raise ValueError("computed failure must use an operational failure status")


type ComputedOutcome[ResultT: ContractModel] = (
    ComputedSuccess[ResultT] | ComputedNotApplicable | OperationExecutionFailure
)


@dataclass(frozen=True, slots=True)
class ComputedOperation[
    RequestT: ContractModel,
    ResultT: ContractModel,
]:
    """One deterministic finite producer capped at computed assurance."""

    capability_id: str
    title: str
    description: str
    request_model: type[RequestT]
    result_model: type[ResultT]
    implementation: Callable[[RequestT], ComputedOutcome[ResultT]]
    relation_id: str
    tags: tuple[str, ...] = ()
    invalid_request: CapabilityDiagnostic | None = None
    invocation_examples: tuple[CapabilityInvocationExample, ...] = ()
    provider_runtime: CapabilityProviderRuntime | None = None
    version: str = "2"


@dataclass(frozen=True, slots=True)
class MaterializedOperation[
    RequestT: ContractModel,
    ArtifactT: ContractModel,
    PreviewT: ContractModel,
    PayloadT: ContractModel,
]:
    """One deterministic producer whose durable result is materialized."""

    capability_id: str
    title: str
    description: str
    request_model: type[RequestT]
    result_model: type[ArtifactT]
    implementation: Callable[[RequestT], ComputedOutcome[ArtifactT]]
    relation_id: str
    tags: tuple[str, ...] = ()
    invalid_request: CapabilityDiagnostic | None = None
    invocation_examples: tuple[CapabilityInvocationExample, ...] = ()
    preview_model: type[PreviewT] | None = None
    preview: Callable[[ArtifactT], PreviewT] | None = None
    preview_complete: bool = False
    accepted_result_capability_ids: tuple[str, ...] = ()
    artifact_converter: (
        Callable[[RequestT, PayloadT], tuple[RequestT, tuple[str, ...]]] | None
    ) = None
    artifact_payload_model: type[PayloadT] | None = None
    artifact_uri_field: str | None = None
    resource_reason: str = ""
    provider_runtime: CapabilityProviderRuntime | None = None
    version: str = "2"

    def __post_init__(self) -> None:
        if not self.resource_reason.strip():
            raise ValueError(
                "materialized operations must declare an explicit resource reason"
            )
        if self.preview_complete and self.preview is None:
            raise ValueError("a complete materialized preview requires a preview")
        has_converter = self.artifact_converter is not None
        if bool(self.accepted_result_capability_ids) != has_converter:
            raise ValueError(
                "accepted result capabilities and an artifact converter must be "
                "declared together"
            )
        if has_converter and (
            self.artifact_payload_model is None or self.artifact_uri_field is None
        ):
            raise ValueError(
                "an artifact converter requires a payload model and URI field"
            )


@dataclass(frozen=True, slots=True)
class OperationFailure:
    """Fail-closed mapping for expected mathematical-domain errors."""

    code: str
    stage: str
    hint: str
    exceptions: tuple[type[Exception], ...] = (TypeError, ValueError)

    def diagnostic(self, error: Exception) -> CapabilityDiagnostic:
        return CapabilityDiagnostic(
            code=self.code,
            stage=self.stage,
            message=str(error),
            hint=self.hint,
        )


@dataclass(frozen=True, slots=True)
class ComputedOperationFactory:
    """Build computed operations with one domain error policy."""

    failure: OperationFailure

    def __call__[
        RequestT: ContractModel,
        ResultT: ContractModel,
    ](
        self,
        capability_id: str,
        title: str,
        description: str,
        request_model: type[RequestT],
        result_model: type[ResultT],
        operation: Callable[[RequestT], ResultT],
        *tags: str,
        invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
        relation_id: str | None = None,
        provider_runtime: CapabilityProviderRuntime | None = None,
    ) -> ComputedOperation[RequestT, ResultT]:
        def implementation(request: RequestT) -> ComputedOutcome[ResultT]:
            try:
                return ComputedSuccess(operation(request))
            except self.failure.exceptions as exc:
                return ComputedNotApplicable(self.failure.diagnostic(exc))

        return ComputedOperation(
            capability_id=capability_id,
            title=title,
            description=description,
            request_model=request_model,
            result_model=result_model,
            implementation=implementation,
            relation_id=relation_id or _relation_id(capability_id),
            tags=tags,
            invocation_examples=invocation_examples,
            provider_runtime=provider_runtime,
        )


@dataclass(frozen=True, slots=True)
class MaterializedOperationFactory:
    """Build materialized operations with one domain error policy."""

    failure: OperationFailure

    def __call__[
        RequestT: ContractModel,
        ResultT: ContractModel,
    ](
        self,
        capability_id: str,
        title: str,
        description: str,
        request_model: type[RequestT],
        result_model: type[ResultT],
        operation: Callable[[RequestT], ResultT],
        *tags: str,
        invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
        relation_id: str | None = None,
        resource_reason: str = "",
        provider_runtime: CapabilityProviderRuntime | None = None,
        preview: Callable[[ResultT], ResultT] | None = None,
        preview_complete: bool = False,
        version: str = "2",
    ) -> MaterializedOperation[RequestT, ResultT, ResultT, ContractModel]:
        def implementation(request: RequestT) -> ComputedOutcome[ResultT]:
            try:
                return ComputedSuccess(operation(request))
            except self.failure.exceptions as exc:
                return ComputedNotApplicable(self.failure.diagnostic(exc))

        return MaterializedOperation(
            capability_id=capability_id,
            title=title,
            description=description,
            request_model=request_model,
            result_model=result_model,
            implementation=implementation,
            relation_id=relation_id or _relation_id(capability_id),
            tags=tags,
            invocation_examples=invocation_examples,
            resource_reason=resource_reason,
            provider_runtime=provider_runtime,
            preview_model=result_model,
            preview=preview,
            preview_complete=preview_complete,
            version=version,
        )


def _relation_id(capability_id: str) -> str:
    segments = capability_id.split(".")
    for index, segment in enumerate(segments):
        if segment in {
            "classify",
            "compute",
            "count",
            "decide",
            "enumerate",
            "evaluate",
            "materialize",
            "solve",
            "transform",
        }:
            segments[index] = "relation"
            return ".".join(segments)
    raise ValueError(
        f"capability ID has no supported operation segment: {capability_id}"
    )


@dataclass(frozen=True, slots=True)
class BoundedSearchWitness[ResultT: ContractModel]:
    """Incumbent/witness found within the declared wall budget."""

    value: ResultT


@dataclass(frozen=True, slots=True)
class BoundedSearchIncomplete[ResultT: ContractModel]:
    """Contract-valid partial result that carries no conclusion."""

    value: ResultT


@dataclass(frozen=True, slots=True)
class BoundedSearchInterrupted[ResultT: ContractModel]:
    """Inspectable partial result from an interrupted bounded execution."""

    value: ResultT
    status: ExecutionStatus
    diagnostic: CapabilityDiagnostic

    def __post_init__(self) -> None:
        if self.status not in {
            ExecutionStatus.ERROR,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }:
            raise ValueError(
                "bounded-search interruption must use an operational failure status"
            )


@dataclass(frozen=True, slots=True)
class BoundedSearchNotApplicable:
    """Valid request outside an operation's mathematical domain."""

    diagnostic: CapabilityDiagnostic


type BoundedSearchOutcome[ResultT: ContractModel] = (
    BoundedSearchWitness[ResultT]
    | BoundedSearchIncomplete[ResultT]
    | BoundedSearchInterrupted[ResultT]
    | BoundedSearchNotApplicable
    | OperationExecutionFailure
)


@dataclass(frozen=True, slots=True)
class BoundedSearchOperation[
    RequestT: ContractModel,
    ResultT: ContractModel,
    ObligationT: ContractModel,
]:
    """One budgeted producer with explicit partial-result semantics."""

    capability_id: str
    title: str
    description: str
    request_model: type[RequestT]
    result_model: type[ResultT]
    implementation: Callable[[RequestT], BoundedSearchOutcome[ResultT]]
    relation_id: str
    scope_parameters: Callable[[RequestT, ResultT], dict[str, Any]]
    is_complete: Callable[[ResultT], bool]
    obligation_model: type[ObligationT]
    obligation: Callable[[RequestT, ResultT], ObligationT]
    incomplete_basis: str
    tags: tuple[str, ...] = ()
    invocation_examples: tuple[CapabilityInvocationExample, ...] = ()
    invalid_request: CapabilityDiagnostic | None = None
    provider_runtime: CapabilityProviderRuntime | None = None
    version: str = "1"


@dataclass(frozen=True, slots=True)
class DomainSemantics:
    """Content-addressed mathematical semantics."""

    name: str
    version: str
    definition: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DomainDiagnostics:
    """Domain wording for request-boundary failures."""

    invalid_request: CapabilityDiagnostic


type DomainOperation = (
    ComputedOperation[Any, Any]
    | MaterializedOperation[Any, Any, Any, Any]
    | BoundedSearchOperation[Any, Any, Any]
)


class ManagedDomainInstaller(Protocol):
    """Install a domain adapter whose resource policy is not generic."""

    def __call__(
        self,
        context: InstallationContext,
        dependencies: Mapping[str, InstalledDomainBundle],
    ) -> InstalledDomainBundle: ...


@dataclass(frozen=True, slots=True)
class DomainBundle:
    """Explicit installation unit owned by one mathematical domain."""

    domain_id: str
    schema_namespace: str
    semantics: DomainSemantics
    provider_runtime: CapabilityProviderRuntime
    backend_version: str
    capabilities: tuple[DomainOperation, ...]
    diagnostics: DomainDiagnostics
    scope_description: str
    completeness_basis: str
    assurance_basis: str
    managed_capability_ids: tuple[str, ...] = ()
    managed_installer: ManagedDomainInstaller | None = None
    dependency_ids: tuple[str, ...] = ()
    checker_declarations: tuple[ExactReplayCheckerDeclaration, ...] = ()

    @property
    def capability_ids(self) -> tuple[str, ...]:
        """Return the capability IDs declared by exactly one installation mode."""

        if self.managed_installer is not None:
            return self.managed_capability_ids
        return tuple(operation.capability_id for operation in self.capabilities)
