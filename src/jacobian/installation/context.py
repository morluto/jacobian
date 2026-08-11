"""Narrow foundational dependencies shared by capability installers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityAdapter, CapabilityService
from jacobian.contracts.capabilities import CapabilityCatalogRelationship
from jacobian.operation_installation import OperationInstaller
from jacobian.registry import CheckerRegistry
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.config import RuntimeOptions
    from jacobian.runtime.services import ApplicationServices, CoreServices


@dataclass(frozen=True, slots=True)
class InstallationContext:
    """Infrastructure used across independent domain installers."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    capabilities: CapabilityService
    checkers: CheckerRegistry
    verification: VerificationService
    operations: OperationInstaller
    checker_authority: CheckerAuthorityMode
    register_capability: Callable[[CapabilityAdapter], None]
    register_checker_relationship: Callable[[str, CapabilityCatalogRelationship], None]

    @property
    def authorizes_bundled_checkers(self) -> bool:
        """Whether built-in checker declarations may be authorized."""

        return self.checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED


def create_installation_context(
    core: CoreServices,
    application: ApplicationServices,
    options: RuntimeOptions,
) -> InstallationContext:
    """Build the production installation context for one service graph.

    Installation contexts are deliberately derived from the explicit core and
    application graphs.  Keeping this wiring here gives domain and composition
    callers one production seam while preserving capability exclusions and the
    operator-owned checker authority configured on ``RuntimeOptions``.

    The registrar is the only place where capability exclusions are applied;
    installers can therefore remain independent of runtime configuration while
    every adapter (including adapters loaded from an entrypoint) follows the
    same policy.
    """

    if application.core is not core:
        raise ValueError("application services must be built from the supplied core")

    excluded = options.capability_exclusions

    def register(adapter: CapabilityAdapter) -> None:
        if adapter.descriptor.capability_id not in excluded:
            core.capabilities.register(adapter)

    def register_checker_relationship(
        source_capability_id: str,
        relationship: CapabilityCatalogRelationship,
    ) -> None:
        core.capabilities._register_catalog_relationship(
            source_capability_id, relationship
        )

    return InstallationContext(
        store=core.store,
        schemas=core.schemas,
        artifacts=core.artifacts,
        capabilities=core.capabilities,
        checkers=core.checkers,
        verification=application.verification,
        operations=core.operations,
        checker_authority=options.checker_authority,
        register_capability=register,
        register_checker_relationship=register_checker_relationship,
    )
