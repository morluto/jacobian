"""Typed operations and bundles owned by mathematical domains."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from jacobian.checker_operations import ExactReplayCheckerDeclaration
    from jacobian.installation.context import InstallationContext
    from jacobian.operation_bindings import InstalledOperation
    from jacobian.operation_installation import InstalledDomainBundle

from jacobian.contracts.capabilities import (
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus


@dataclass(frozen=True, slots=True)
class NonConclusion:
    """Expected refusal or interruption carrying no mathematical conclusion."""

    diagnostic: CapabilityDiagnostic


@dataclass(frozen=True, slots=True)
class Failed:
    """Operational failure that carries no mathematical conclusion."""

    status: ExecutionStatus
    diagnostic: CapabilityDiagnostic

    def __post_init__(self) -> None:
        if self.status not in {
            ExecutionStatus.ERROR,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }:
            raise ValueError("failed operation must use an operational failure status")


@dataclass(frozen=True, slots=True)
class Completed[ResultT: ContractModel]:
    """Contract-valid mathematical result ready for publication."""

    value: ResultT
    runtime_ms: int


class Effect(StrEnum):
    """Semantic effect of executing an operation, independent of publication."""

    READ_ONLY = "READ_ONLY"
    STATEFUL = "STATEFUL"


class PreflightStatus(StrEnum):
    """Bounded admission result evaluated before mathematical execution."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Small typed preflight decision with an optional stable reason."""

    status: PreflightStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is PreflightStatus.SUPPORTED and self.reason is not None:
            raise ValueError("supported preflight cannot carry a rejection reason")
        if self.status is not PreflightStatus.SUPPORTED and not (
            self.reason and self.reason.strip()
        ):
            raise ValueError("rejected preflight requires a reason")


SUPPORTED = PreflightResult(PreflightStatus.SUPPORTED)


@dataclass(frozen=True, slots=True)
class OperationSpec[
    RequestT: ContractModel,
    ResultT: ContractModel,
]:
    """Semantic declaration for one deterministic mathematical operation."""

    operation_id: str
    version: str
    request_type: type[RequestT]
    result_type: type[ResultT]
    execute: Callable[[RequestT], ResultT]
    title: str
    description: str
    tags: tuple[str, ...] = ()
    invalid_request: CapabilityDiagnostic | None = None
    invocation_examples: tuple[CapabilityInvocationExample, ...] = ()
    preflight: Callable[[RequestT], PreflightResult] | None = None
    postcondition: Callable[[RequestT, ResultT], None] | None = None
    effect: Effect = Effect.READ_ONLY


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


class OperationRefusalError(Exception):
    """Signal an expected domain refusal without returning terminal state."""

    def __init__(self, diagnostic: CapabilityDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class OperationAbortError(Exception):
    """Signal an operational interruption without returning terminal state."""

    def __init__(
        self,
        status: ExecutionStatus,
        diagnostic: CapabilityDiagnostic,
    ) -> None:
        if status not in {
            ExecutionStatus.ERROR,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }:
            raise ValueError("aborted operation must use an operational failure status")
        super().__init__(diagnostic.message)
        self.status = status
        self.diagnostic = diagnostic


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
    capabilities: tuple[InstalledOperation[Any, Any], ...]
    diagnostics: DomainDiagnostics
    managed_capability_ids: tuple[str, ...] = ()
    managed_installer: ManagedDomainInstaller | None = None
    dependency_ids: tuple[str, ...] = ()
    checker_declarations: tuple[ExactReplayCheckerDeclaration, ...] = ()

    @property
    def capability_ids(self) -> tuple[str, ...]:
        """Return the capability IDs declared by exactly one installation mode."""

        if self.managed_installer is not None:
            return self.managed_capability_ids
        return tuple(operation.spec.operation_id for operation in self.capabilities)
