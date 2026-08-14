"""Operator-only construction of built-in catalog descriptors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jacobian.artifacts import ArtifactService
from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.checker_identity import batch_checker_manifest_measurement
from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    ExactOperationGroup,
)
from jacobian.exact_domain_verification import (
    install_exact_domain_verification,
)
from jacobian.implementation import cached_package_digests
from jacobian.lean_frontend.declarations import LeanDeclarationService
from jacobian.lean_frontend.exploration import LeanExplorationInstallation
from jacobian.lean_frontend.service import LeanService
from jacobian.maintained_backends import require_maintained_math_backends
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.selected_families import (
    selected_family_catalog_installers,
    selected_family_specs,
)
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.value_references import ValueReferenceStore
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.resources import RuntimeResources


@dataclass(frozen=True, slots=True)
class CatalogBuildContext:
    """Resources required while compiling the explicit built-in catalog."""

    store: ArtifactRepository
    schemas: SchemaRegistry
    artifacts: ArtifactService
    values: ValueReferenceStore
    checkers: CheckerRegistry
    verification: VerificationService
    binder: OperationBinder
    sat: SatArtifactService
    smt: SmtArtifactService
    polynomial_expressions: PolynomialExpressionArtifactService
    authorize_bundled_checkers: bool
    register_operation: Callable[[OperationAdapter[Any]], None]


@dataclass(slots=True)
class CatalogBuildResources:
    """Closeable resources retained during operator catalog compilation."""

    lean: LeanService | None = None
    lean_declarations: LeanDeclarationService | None = None
    lean_exploration: LeanExplorationInstallation | None = None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        for resource in (
            self.lean_declarations,
            self.lean_exploration.repl if self.lean_exploration is not None else None,
            self.lean,
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            exceptions = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exceptions) == len(failures):
                raise ExceptionGroup("catalog resources failed to close", exceptions)
            raise BaseExceptionGroup("catalog resources failed to close", failures)
        self._closed = True


@dataclass(frozen=True, slots=True)
class CatalogOperationBuilder:
    """Build declaration-module operations and exact-domain checkers."""

    context: CatalogBuildContext

    def bind(self) -> None:
        """Load builtin declaration modules and bind exact-domain verification."""

        ctx = self.context
        exact_groups: dict[str, ExactOperationGroup] = {}
        for (
            module_name,
            operations,
            checker_declarations,
        ) in load_builtin_operation_modules():
            bound = ctx.binder.bind(operations)
            for adapter in bound.adapters:
                ctx.register_operation(adapter)
            if checker_declarations:
                exact_groups[module_name] = (
                    operations,
                    bound,
                    checker_declarations,
                )
        self.bind_domain_verification(exact_groups)

    def bind_domain_verification(
        self,
        operation_groups: dict[str, ExactOperationGroup],
    ) -> ExactDomainCheckerInstallation | None:
        ctx = self.context
        if not operation_groups:
            return None
        # Batch identity material across the complete declaration set while the
        # exact-domain installer resolves both legacy and declaration-owned
        # provider runtimes. Nested measurement remains safe for direct callers.
        with batch_checker_manifest_measurement():
            adapters, installation = install_exact_domain_verification(
                ctx.store,
                ctx.schemas,
                ctx.artifacts,
                ctx.values,
                ctx.verification,
                ctx.checkers,
                groups=operation_groups,
                authorize=ctx.authorize_bundled_checkers,
            )
        for adapter in adapters:
            self.context.register_operation(adapter)
        return installation


def create_catalog_build_context(
    core: RuntimeResources,
    verification: VerificationService,
    *,
    authorize_bundled_checkers: bool = False,
) -> CatalogBuildContext:
    """Build the operator-only context for one catalog compilation.

    The context is derived from explicit core and runtime ownership. It keeps
    operation visibility and operator-owned checker authority at the one
    catalog build boundary.

    The registrar is the only place where operation exclusions are applied;
    every built-in adapter therefore follows the same policy.
    """

    def register(adapter: OperationAdapter[Any]) -> None:
        if core.operations is None:
            raise RuntimeError("catalog compilation requires an operation collector")
        core.operations.register(adapter)

    if core.sat is None or core.smt is None or core.polynomial_expressions is None:
        core.ensure_family_artifacts()
    sat = core.sat
    smt = core.smt
    polynomial_expressions = core.polynomial_expressions
    if sat is None or smt is None or polynomial_expressions is None:
        raise RuntimeError("catalog compilation requires family artifact contracts")

    return CatalogBuildContext(
        store=core.store,
        schemas=core.schemas,
        artifacts=core.artifacts,
        values=core.values,
        checkers=core.checkers,
        verification=verification,
        binder=core.binder,
        sat=sat,
        smt=smt,
        polynomial_expressions=polynomial_expressions,
        authorize_bundled_checkers=authorize_bundled_checkers,
        register_operation=register,
    )


def build_catalog_operations(
    context: CatalogBuildContext,
    polytope: PolytopeService,
) -> CatalogBuildResources:
    """Build every descriptor and checker binding in deterministic phase order.

    This function is the single build boundary for the built-in catalog. It
    owns both the ordering of compilation phases and the durable
    transaction that couples operation/checker registration to store writes.
    The checker-policy lock is acquired before the SQLite transaction, as
    required by :class:`CheckerRegistry`, and package digests are cached for
    the duration of the same atomic assembly.
    """

    resources = CatalogBuildResources()
    try:
        with (
            context.checkers.policy_transaction(),
            context.store.transaction(),
            cached_package_digests(),
        ):
            require_maintained_math_backends()
            CatalogOperationBuilder(context).bind()
            installers = selected_family_catalog_installers()
            for spec in selected_family_specs():
                installers[spec.origin](
                    context,
                    polytope=polytope,
                    resources=resources,
                )
    except BaseException as exc:
        try:
            resources.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"partial catalog cleanup also failed: {cleanup_exc}")
        raise
    return resources


__all__ = [
    "CatalogBuildContext",
    "CatalogBuildResources",
    "CatalogOperationBuilder",
    "build_catalog_operations",
    "create_catalog_build_context",
]
