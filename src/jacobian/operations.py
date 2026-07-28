"""Typed operations and bundles owned by mathematical domains."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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
    version: str = "1"


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
        operation: Callable[[ContractModel], ContractModel],
        *tags: str,
        invocation_examples: tuple[CapabilityInvocationExample, ...] = (),
        relation_id: str | None = None,
    ) -> ComputedOperation[RequestT, ResultT]:
        def implementation(request: RequestT) -> ComputedOutcome[ResultT]:
            try:
                return ComputedSuccess(cast(ResultT, operation(request)))
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
        )


def _relation_id(capability_id: str) -> str:
    for verb in ("compute", "decide", "enumerate", "solve", "transform"):
        marker = f".{verb}."
        if marker in capability_id:
            return capability_id.replace(marker, ".relation.", 1)
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
    obligation_model: type[ContractModel]
    obligation: Callable[[RequestT, ResultT], ContractModel]
    incomplete_basis: str
    tags: tuple[str, ...] = ()
    invalid_request: CapabilityDiagnostic | None = None
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


type DomainOperation = ComputedOperation[Any, Any] | BoundedSearchOperation[Any, Any]


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
