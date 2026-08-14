"""Operator-controlled declarations for independent exact-operation replay."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jacobian.checker_authorization import authorize_checker_operation
from jacobian.checker_identity import batch_checker_manifest_measurement
from jacobian.checker_operations import AuthorizedChecker, CheckerOperation
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.operations import (
    OperationDiagnostic,
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.operation_binding import BoundOperationGroup
from jacobian.operation_declarations import (
    InlineOperation,
    OperationDeclaration,
    OperationDeclarations,
)
from jacobian.providers.flint_runtime import (
    exact_domain_checker_source_provider_runtime,
)
from jacobian.registry import CheckerRegistry

_LOGGER = logging.getLogger(__name__)

type ExactOperationGroup = tuple[
    OperationDeclarations,
    BoundOperationGroup,
    tuple[AuthorizedChecker, ...],
]


@dataclass(frozen=True, slots=True)
class ExactDomainCheckerInstallation:
    """Exact replay identities and non-conclusive installation diagnostics."""

    checker_ids: dict[str, str | None]
    provider_runtimes: dict[str, ProviderObservation]
    declaration_providers: dict[str, str]
    witness_schema_uri: str | None = None
    diagnostics: tuple[OperationDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class _DeclaredRuntimeGroup:
    probe: ProviderObservation
    members: tuple[tuple[BoundOperationGroup, AuthorizedChecker], ...]


def _declared_runtime_groups(
    pairs: tuple[tuple[BoundOperationGroup, AuthorizedChecker], ...],
) -> tuple[_DeclaredRuntimeGroup, ...]:
    grouped: dict[
        str,
        tuple[
            ProviderObservation,
            list[tuple[BoundOperationGroup, AuthorizedChecker]],
        ],
    ] = {}
    for installed, declaration in pairs:
        probe = declaration.observation_loader()
        if not isinstance(probe, ProviderObservation):
            raise TypeError(
                "checker observation loader must return ProviderObservation"
            )
        if probe.checker_ids:
            raise ValueError("checker declarations must not pre-authorize checker IDs")
        current = grouped.get(probe.provider)
        if current is None:
            grouped[probe.provider] = (probe, [(installed, declaration)])
            continue
        existing_probe, members = current
        if existing_probe.model_dump(mode="json") != probe.model_dump(mode="json"):
            raise ValueError(
                "exact replay grouped distinct probes under one provider "
                f"identity: {probe.provider}"
            )
        members.append((installed, declaration))
    return tuple(
        _DeclaredRuntimeGroup(
            probe=probe,
            members=tuple(members),
        )
        for probe, members in grouped.values()
    )


def _authorize_replay_operation(
    checkers: CheckerRegistry,
    operation: CheckerOperation,
    *,
    authorize: bool,
    provider_runtime: ProviderObservation,
    source_available: bool,
    optional: bool,
    operation_id: str,
    diagnostics: list[OperationDiagnostic],
) -> str | None:
    if provider_runtime.availability is ProviderAvailability.AVAILABLE:
        return authorize_checker_operation(
            checkers, operation, authorize=authorize
        ).checker_id
    can_omit = optional and source_available
    if not can_omit:
        return authorize_checker_operation(
            checkers, operation, authorize=authorize
        ).checker_id
    diagnostic = OperationDiagnostic(
        code="EXACT_REPLAY_PROVIDER_UNAVAILABLE",
        stage="provider_availability",
        message=(
            f"Independent replay for {operation_id!r} is not installed: "
            f"{provider_runtime.diagnostic or 'the provider is unavailable.'}"
        ),
        hint="Install or repair the optional python-flint backend, then retry.",
        details={
            "operation_id": operation_id,
            "provider": provider_runtime.provider,
            "checker_authorization_affected": True,
        },
    )
    diagnostics.append(diagnostic)
    _LOGGER.warning("%s", diagnostic.message)
    return None


def _authorized_provider_runtimes(
    groups: tuple[_DeclaredRuntimeGroup, ...],
    checker_ids: Mapping[str, str | None],
) -> dict[str, ProviderObservation]:
    provider_runtimes: dict[str, ProviderObservation] = {}
    for group in groups:
        authorized = tuple(
            checker_id
            for _installed, declaration in group.members
            if (checker_id := checker_ids[declaration.operation_id]) is not None
        )
        runtime = group.probe.model_copy(update={"checker_ids": authorized})
        existing = provider_runtimes.get(runtime.provider)
        if existing is not None and existing.model_dump(
            mode="json"
        ) != runtime.model_dump(mode="json"):
            raise ValueError(
                "exact replay grouped distinct runtimes under one provider "
                f"identity: {runtime.provider}"
            )
        if runtime.provider != group.probe.provider:
            raise ValueError(
                "exact replay authorized runtime changed provider identity: "
                f"{group.probe.provider} -> {runtime.provider}"
            )
        provider_runtimes[runtime.provider] = runtime
    return provider_runtimes


def _available_declaration_groups(
    groups: Mapping[str, ExactOperationGroup],
) -> tuple[tuple[BoundOperationGroup, AuthorizedChecker], ...]:
    """Pair checker declarations with their unique bound producer."""

    available: list[tuple[BoundOperationGroup, AuthorizedChecker]] = []
    owners: dict[str, str] = {}
    for module_name, (operations, installed, declarations) in groups.items():
        producer_operation_ids = {
            _operation_spec(operation).operation_id for operation in operations
        }
        installed_producer_ids = {
            adapter.descriptor.operation_id for adapter in installed.adapters
        }
        for declaration in declarations:
            if declaration.operation_id not in producer_operation_ids:
                raise ValueError(
                    "exact replay declaration is not backed by a domain producer "
                    f"schema: {module_name}/{declaration.operation_id}"
                )
            if declaration.operation_id not in installed.result_schema_uris:
                continue
            if (
                installed.adapters
                and declaration.operation_id not in installed_producer_ids
            ):
                continue
            previous = owners.setdefault(declaration.operation_id, module_name)
            if previous != module_name:
                raise ValueError(
                    "exact replay declaration is owned by multiple groups: "
                    f"{declaration.operation_id}"
                )
            available.append((installed, declaration))
    operation_ids = [declaration.operation_id for _, declaration in available]
    if len(operation_ids) != len(set(operation_ids)):
        duplicates = sorted(
            operation_id
            for operation_id in set(operation_ids)
            if operation_ids.count(operation_id) > 1
        )
        if duplicates:
            raise ValueError(
                "operation group repeats exact replay declarations: "
                + ", ".join(duplicates)
            )
    return tuple(available)


def install_exact_domain_checkers(
    checkers: CheckerRegistry,
    *,
    groups: Mapping[str, ExactOperationGroup],
    authorize: bool,
) -> ExactDomainCheckerInstallation:
    """Install independent exact replay against dynamically registered schemas."""

    available_declarations = _available_declaration_groups(groups)
    checker_ids: dict[str, str | None] = {}
    declaration_providers: dict[str, str] = {}
    diagnostics: list[OperationDiagnostic] = []
    checker_ids.update(
        (declaration.operation_id, None)
        for _installed, declaration in available_declarations
    )
    if not authorize and not checkers.bind_existing_when_omitted:
        return ExactDomainCheckerInstallation(
            checker_ids=checker_ids,
            provider_runtimes={},
            declaration_providers={},
        )
    source_available = (
        exact_domain_checker_source_provider_runtime().availability
        is ProviderAvailability.AVAILABLE
    )
    with batch_checker_manifest_measurement():
        runtime_groups = _declared_runtime_groups(available_declarations)
        for group in runtime_groups:
            for installed, declaration in group.members:
                declaration_providers[declaration.operation_id] = group.probe.provider
                operation = CheckerOperation(
                    name=(
                        f"{declaration.operation_id} independent "
                        f"{declaration.replay_method}"
                    ),
                    entrypoint=(
                        f"{declaration.entrypoint_module}:{declaration.function}"
                    ),
                    evidence_kind=EvidenceKind.WITNESS,
                    format_id=declaration.format_id,
                    format_version="1",
                    claim_schema_uris=(
                        installed.input_schema_uris[declaration.request_model],
                    ),
                    semantics_uris=(installed.semantics_uri,),
                    candidate_schema_uris=(
                        installed.result_schema_uris[declaration.operation_id],
                    ),
                    reason=declaration.reason,
                    provider_runtime=group.probe,
                )
                checker_ids[declaration.operation_id] = _authorize_replay_operation(
                    checkers,
                    operation,
                    authorize=authorize,
                    provider_runtime=group.probe,
                    source_available=source_available,
                    optional=declaration.optional,
                    operation_id=declaration.operation_id,
                    diagnostics=diagnostics,
                )
    return ExactDomainCheckerInstallation(
        checker_ids=checker_ids,
        diagnostics=tuple(diagnostics),
        provider_runtimes=_authorized_provider_runtimes(runtime_groups, checker_ids),
        declaration_providers=declaration_providers,
    )


__all__ = [
    "ExactDomainCheckerInstallation",
    "install_exact_domain_checkers",
]


def _operation_spec(operation: Any) -> Any:
    if isinstance(operation, (InlineOperation, OperationDeclaration)):
        return operation
    return operation.spec
