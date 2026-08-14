"""Dependencies owned by built-in mathematical portfolio assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jacobian.artifacts import ArtifactService
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
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
class PortfolioContext:
    """Resources required while assembling the explicit built-in portfolio."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    checkers: CheckerRegistry
    verification: VerificationService
    binder: OperationBinder
    checker_authority: CheckerAuthorityMode
    register_operation: Callable[[OperationAdapter[Any]], None]

    @property
    def authorizes_bundled_checkers(self) -> bool:
        """Whether built-in checker declarations may be authorized."""

        return self.checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED


def create_portfolio_context(
    core: CoreServices,
    services: RuntimeServices,
    options: RuntimeOptions,
) -> PortfolioContext:
    """Build the portfolio assembly context for one runtime graph.

    The context is derived from explicit core and runtime ownership. It keeps
    operation visibility and operator-owned checker authority at the one
    portfolio composition boundary.

    The registrar is the only place where operation exclusions are applied;
    every built-in adapter therefore follows the same policy.
    """

    if services.core is not core:
        raise ValueError("runtime services must be built from the supplied core")

    excluded = options.operation_exclusions

    def register(adapter: OperationAdapter[Any]) -> None:
        if adapter.descriptor.operation_id not in excluded:
            core.operations.register(adapter)

    return PortfolioContext(
        store=core.store,
        schemas=core.schemas,
        artifacts=core.artifacts,
        values=core.values,
        checkers=core.checkers,
        verification=services.verification,
        binder=core.binder,
        checker_authority=options.checker_authority,
        register_operation=register,
    )
