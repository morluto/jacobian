"""Construct a catalog-backed runtime for selected operation execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from jacobian.operation_catalog import OperationCatalog, OperationCatalogView
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.operation_registry import OperationRegistry
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.package_index import PackageIndexRegistry
from jacobian.polytope import PolytopeService
from jacobian.registry import CheckerRegistry
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import InlineServingResources, JacobianRuntime
from jacobian.runtime.resources import RuntimeResources
from jacobian.runtime.selected_families import create_runtime_selected_families
from jacobian.selected_operation_bindings import RuntimeSelectedFamily
from jacobian.serving_catalog import ServingCatalog
from jacobian.verification.service import VerificationService


@dataclass(slots=True)
class LazyControlPlane:
    """Construct verification, polytope, and family binders on first non-inline use."""

    core: RuntimeResources
    catalog: OperationCatalog
    _lock: Lock = field(default_factory=Lock)
    _verification: VerificationService | None = None
    _polytope: PolytopeService | None = None
    _families: tuple[RuntimeSelectedFamily, ...] | None = None

    def materialize(self) -> None:
        if self._families is not None:
            return
        with self._lock:
            if self._families is not None:
                return
            self.core.ensure_family_artifacts()
            sat = self.core.sat
            smt = self.core.smt
            polynomial_expressions = self.core.polynomial_expressions
            if sat is None or smt is None or polynomial_expressions is None:
                raise RuntimeError("family artifact contracts were not installed")
            verification = VerificationService(
                self.core.store,
                self.core.checkers,
                self.core.schemas,
                checker_timeout_seconds=105,
            )
            polytope = PolytopeService(self.core.store, self.core.schemas)
            self._verification = verification
            self._polytope = polytope
            self._families = create_runtime_selected_families(
                catalog=self.catalog,
                binder=self.core.binder,
                verification=verification,
                checkers=self.core.checkers,
                polynomial_expressions=polynomial_expressions,
                polytope=polytope,
                sat=sat,
                smt=smt,
                runtime_resources=self.core,
            )

    @property
    def verification(self) -> VerificationService:
        self.materialize()
        assert self._verification is not None
        return self._verification

    @property
    def polytope(self) -> PolytopeService:
        self.materialize()
        assert self._polytope is not None
        return self._polytope

    @property
    def families(self) -> tuple[RuntimeSelectedFamily, ...]:
        self.materialize()
        assert self._families is not None
        return self._families


def create_inline_serving_runtime(
    catalog: ServingCatalog,
) -> JacobianRuntime:
    """Serve packaged inline operations without opening a state directory.

    Family IDs remain discoverable. ``PackageIndexRegistry`` resolves them to a
    structured ``STATE_INITIALIZATION_REQUIRED`` failure instead of loading
    them as inline symbols.
    """

    registry = PackageIndexRegistry(catalog.index)
    dispatcher = OperationDispatcher(catalog, registry)
    return JacobianRuntime(
        InlineServingResources(dispatcher),
        inline_serving=True,
    )


def create_execution_runtime(
    root: str | Path,
    catalog: OperationCatalogView,
    *,
    operation_policy: OperationVisibilityPolicy,
    checker_registry: CheckerRegistry | None = None,
) -> JacobianRuntime:
    """Open artifact state and defer family machinery until a non-inline ID."""

    sqlite = catalog.overlay if isinstance(catalog, ServingCatalog) else catalog
    if not isinstance(sqlite, OperationCatalog):
        raise TypeError("execution runtime requires a SQLite operation catalog")
    core = bootstrap_services(
        root,
        operation_policy=operation_policy,
        bind_existing_checkers=True,
        install_family_artifacts=False,
        collect_operations=False,
    )
    try:
        if checker_registry is not None:
            core.checkers = checker_registry
        control_plane = LazyControlPlane(core, sqlite)
        registry = OperationRegistry(
            catalog,
            core.binder,
            core.checkers,
            core,
            control_plane=control_plane,
            package_index=catalog.index
            if isinstance(catalog, ServingCatalog)
            else None,
        )
        core.operations = OperationDispatcher(catalog, registry)
        return JacobianRuntime(core, control_plane=control_plane)
    except BaseException as exc:
        try:
            core.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"runtime construction cleanup also failed: {cleanup_exc}")
        raise


def create_serving_runtime(
    root: str | Path,
    catalog: ServingCatalog,
    *,
    operation_policy: OperationVisibilityPolicy,
    checker_registry: CheckerRegistry | None = None,
) -> JacobianRuntime:
    """Serve from built-in declarations, opening SQLite only when overlay state exists."""

    if catalog.overlay is None:
        return create_inline_serving_runtime(catalog)
    return create_execution_runtime(
        root,
        catalog,
        operation_policy=operation_policy,
        checker_registry=checker_registry,
    )


__all__ = [
    "LazyControlPlane",
    "create_execution_runtime",
    "create_inline_serving_runtime",
    "create_serving_runtime",
]
