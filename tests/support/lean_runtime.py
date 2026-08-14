"""Lean-only runtimes for behavior tests that must not assemble the portfolio.

Construction uses production Lean installers and checker authorization. It does
not call ``create_catalog_build_runtime`` or bind graph, polynomial, SAT, or
other unrelated families.
"""

from __future__ import annotations

import logging
from pathlib import Path

from filelock import FileLock

from jacobian.builtin_operations import LeanCheckAdapter
from jacobian.catalog.build import (
    CatalogBuildContext,
    create_catalog_build_context,
)
from jacobian.checker_authorization import install_lean_checkers
from jacobian.contracts.operations import ProviderAvailability
from jacobian.lean_frontend.declaration_operations import (
    lean_declaration_query_operations,
)
from jacobian.lean_frontend.declarations import (
    LeanDeclarationService,
    installed_lean_declaration_service,
)
from jacobian.lean_frontend.exploration import install_lean_exploration_operations
from jacobian.lean_frontend.proof_axioms import install_lean_proof_axioms_operation
from jacobian.lean_frontend.proof_edit import install_lean_proof_edit_operation
from jacobian.lean_frontend.proof_state_inspect import (
    install_lean_proof_state_inspect_only,
)
from jacobian.lean_frontend.service import LeanService
from jacobian.lean_frontend.statement import install_lean_statement_operations
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.polytope import PolytopeService
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.providers.lean_runtime import lean_provider_runtime
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import atomic_installation
from tests.support.state import publish_template, quiesce_sqlite_template

_LOGGER = logging.getLogger(__name__)


class LeanRuntime(JacobianRuntime):
    """Selected Lean family plus the owning runtime close boundary."""

    def __init__(
        self,
        core: RuntimeResources,
        verification: VerificationService,
        polytope: PolytopeService,
        *,
        lean: LeanService | None,
        lean_declarations: LeanDeclarationService | None,
    ) -> None:
        super().__init__(core, verification, polytope)
        self.lean = lean
        self.lean_declarations = lean_declarations

    def __enter__(self) -> LeanRuntime:
        super().__enter__()
        return self


def create_lean_runtime(
    root: str | Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.INSTALL_BUNDLED,
    operation_policy: OperationVisibilityPolicy | None = None,
) -> LeanRuntime:
    """Install Lean checkers, declarations, exploration, and statements only."""

    core = bootstrap_services(
        root,
        operation_policy=operation_policy,
        bind_existing_checkers=(
            checker_authority is CheckerAuthorityMode.HYDRATE_EXISTING
        ),
    )
    try:
        verification = VerificationService(
            core.store,
            core.checkers,
            core.schemas,
            checker_timeout_seconds=105,
        )
        polytope = PolytopeService(core.store, core.schemas)
        context = create_catalog_build_context(
            core,
            verification,
            authorize_bundled_checkers=(
                checker_authority is CheckerAuthorityMode.INSTALL_BUNDLED
            ),
        )
        lean: LeanService | None = None
        lean_declarations: LeanDeclarationService | None = None
        bind_lean = (
            context.authorize_bundled_checkers
            or core.checkers.bind_existing_when_omitted
        )
        if bind_lean:
            with atomic_installation(core):
                lean, lean_declarations = _install_lean_family(context, core)
        runtime = LeanRuntime(
            core,
            verification,
            polytope,
            lean=lean,
            lean_declarations=lean_declarations,
        )
        return runtime
    except BaseException as error:
        try:
            core.close()
        except BaseException as cleanup_error:
            error.add_note(
                "Lean runtime construction cleanup also failed: " + str(cleanup_error)
            )
        raise


def _install_lean_family(
    context: CatalogBuildContext,
    core: RuntimeResources,
) -> tuple[LeanService | None, LeanDeclarationService | None]:
    lean_checkers, checker_runtime = install_lean_checkers(
        context.store,
        context.schemas,
        context.checkers,
        resolve_provider_runtime=lambda profiles: lean_provider_runtime(
            profiles=profiles,
            checker_ids=(),
        ),
    )
    runtime = checker_runtime.model_copy(
        update={
            "checker_ids": tuple(
                installation.checker_id
                for _, installation in sorted(
                    lean_checkers.items(),
                    key=lambda item: item[0].value,
                )
                if installation.checker_id is not None
            )
        }
    )
    inspect_adapter = install_lean_proof_state_inspect_only(
        context.store,
        context.schemas,
        context.artifacts,
        lean_checkers,
        jacobian_provider_runtime(
            "jacobian.lean4",
            features=("immutable-proof-state", "read-only-inspection"),
        ),
    )
    context.register_operation(inspect_adapter)
    statement_adapters, _ = install_lean_statement_operations(
        context.store,
        context.schemas,
        context.artifacts,
        runtime,
    )
    for adapter in statement_adapters:
        context.register_operation(adapter)
    if runtime.availability is not ProviderAvailability.AVAILABLE:
        _LOGGER.warning("lean.check is not installed: %s", runtime.diagnostic)
        return None, None
    if any(installation.checker_id is None for installation in lean_checkers.values()):
        _LOGGER.warning("lean.check is not installed: no active Lean checker")
        return None, None
    lean_declarations = None
    try:
        lean_declarations = installed_lean_declaration_service(
            runtime,
            cache_root=context.store.root / "cache" / "lean-declarations",
        )
    except (OSError, RuntimeError) as exc:
        _LOGGER.warning("Lean declaration discovery is not installed: %s", exc)
    if lean_declarations is not None:
        core.own(lean_declarations)
        bound_queries = context.binder.bind(
            lean_declaration_query_operations(lean_declarations)
        )
        for adapter in bound_queries.adapters:
            context.register_operation(adapter)
    lean = LeanService(
        context.store,
        context.artifacts,
        context.verification,
        lean_checkers,
    )
    core.own(lean)
    context.register_operation(LeanCheckAdapter(lean, runtime))
    proof_axioms_adapter, _ = install_lean_proof_axioms_operation(
        context.store,
        context.schemas,
        context.artifacts,
        lean_checkers,
        runtime,
    )
    context.register_operation(proof_axioms_adapter)
    adapters, exploration = install_lean_exploration_operations(
        context.store,
        context.schemas,
        context.artifacts,
        lean_checkers,
        runtime,
    )
    core.own(exploration.repl)
    for adapter in adapters:
        if adapter.descriptor.operation_id == "lean.proof_state.inspect":
            continue
        context.register_operation(adapter)
    proof_edit_adapter, _ = install_lean_proof_edit_operation(
        context.store,
        context.schemas,
        context.artifacts,
        lean,
        runtime,
    )
    context.register_operation(proof_edit_adapter)
    return lean, lean_declarations


def publish_lean_authorized_template(
    target: Path,
    *,
    lock: FileLock | None = None,
) -> Path:
    """Install Lean checkers once and publish an immutable snapshot."""

    def build(staging: Path) -> None:
        runtime = create_lean_runtime(
            staging,
            checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        )
        runtime.close()
        quiesce_sqlite_template(staging)

    return publish_template(target, build, lock=lock)


__all__ = [
    "LeanRuntime",
    "create_lean_runtime",
    "publish_lean_authorized_template",
]
