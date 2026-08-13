"""Narrow foundational dependencies shared by operation installers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jacobian.artifacts import ArtifactService
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_installation import OperationInstaller
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.config import RuntimeOptions
    from jacobian.runtime.services import CoreServices, RuntimeServices


@dataclass(frozen=True, slots=True)
class InstallationContext:
    """Infrastructure used across independent domain installers."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    checkers: CheckerRegistry
    verification: VerificationService
    operations: OperationInstaller
    checker_authority: CheckerAuthorityMode
    register_operation: Callable[[OperationAdapter[Any]], None]

    @property
    def authorizes_bundled_checkers(self) -> bool:
        """Whether built-in checker declarations may be authorized."""

        return self.checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED


def create_installation_context(
    core: CoreServices,
    services: RuntimeServices,
    options: RuntimeOptions,
) -> InstallationContext:
    """Build the production installation context for one service graph.

    Installation contexts are deliberately derived from the explicit core and
    runtime graphs.  Keeping this wiring here gives domain and composition
    callers one production seam while preserving operation exclusions and the
    operator-owned checker authority configured on ``RuntimeOptions``.

    The registrar is the only place where operation exclusions are applied;
    installers can therefore remain independent of runtime configuration while
    every built-in adapter follows the same policy.
    """

    if services.core is not core:
        raise ValueError("runtime services must be built from the supplied core")

    excluded = options.operation_exclusions

    def register(adapter: OperationAdapter[Any]) -> None:
        if adapter.descriptor.operation_id not in excluded:
            core.operations.register(adapter)

    return InstallationContext(
        store=core.store,
        schemas=core.schemas,
        artifacts=core.artifacts,
        values=core.values,
        checkers=core.checkers,
        verification=services.verification,
        operations=core.installer,
        checker_authority=options.checker_authority,
        register_operation=register,
    )
