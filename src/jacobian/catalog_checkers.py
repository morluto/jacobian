"""Installation of operator-authorized Lean and polytope checkers.

This phase authorizes the retained polytope and Lean checker families through
the standalone :mod:`jacobian.checker_authorization` module and then binds
the Lean declaration, exploration, proof-axiom, proof-edit, and proof-state
inspection operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jacobian.builtin_operations import LeanCheckAdapter
from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.checker_authorization import (
    install_lean_checkers,
    install_polytope_checkers,
)
from jacobian.contracts.operations import (
    ProviderAvailability,
)
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
from jacobian.polytope import PolytopeService
from jacobian.provider_runtime import jacobian_provider_runtime
from jacobian.providers.lean_runtime import lean_provider_runtime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CatalogCheckerBuilder:
    """Authorize retained checkers and build their catalog descriptors."""

    context: CatalogBuildContext

    def bind(
        self,
        polytope: PolytopeService,
        resources: CatalogBuildResources,
    ) -> None:
        ctx = self.context
        # INSTALL_BUNDLED creates authority; HYDRATE_EXISTING binds only checker
        # identities already authorized on this store. NONE binds neither.
        if ctx.authorize_bundled_checkers or ctx.checkers.bind_existing_when_omitted:
            self._bind_checkers(polytope, resources)

    def _bind_checkers(
        self,
        polytope: PolytopeService,
        resources: CatalogBuildResources,
    ) -> None:
        ctx = self.context
        install_polytope_checkers(
            ctx.checkers,
            claim_schema_uri=polytope.claim_schema_uri,
            semantics_uri=polytope.semantics_uri,
            point_schema_uri=polytope.point_schema_uri,
        )
        lean_checkers, checker_runtime = install_lean_checkers(
            ctx.store,
            ctx.schemas,
            ctx.checkers,
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
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            lean_checkers,
            jacobian_provider_runtime(
                "jacobian.lean4",
                features=("immutable-proof-state", "read-only-inspection"),
            ),
        )
        ctx.register_operation(inspect_adapter)
        if runtime.availability is not ProviderAvailability.AVAILABLE:
            _LOGGER.warning("lean.check is not installed: %s", runtime.diagnostic)
            return
        if any(
            installation.checker_id is None for installation in lean_checkers.values()
        ):
            _LOGGER.warning("lean.check is not installed: no active Lean checker")
            return
        try:
            resources.lean_declarations = installed_lean_declaration_service(
                runtime,
                cache_root=ctx.store.root / "cache" / "lean-declarations",
            )
        except (OSError, RuntimeError) as exc:
            _LOGGER.warning("Lean declaration discovery is not installed: %s", exc)
        self._bind_lean_declaration_adapters(resources.lean_declarations)
        resources.lean = LeanService(
            ctx.store,
            ctx.artifacts,
            ctx.verification,
            lean_checkers,
        )
        ctx.register_operation(LeanCheckAdapter(resources.lean, runtime))
        proof_axioms_adapter, _ = install_lean_proof_axioms_operation(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            lean_checkers,
            runtime,
        )
        ctx.register_operation(proof_axioms_adapter)
        adapters, resources.lean_exploration = install_lean_exploration_operations(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            lean_checkers,
            runtime,
        )
        for adapter in adapters:
            if adapter.descriptor.operation_id == "lean.proof_state.inspect":
                continue
            ctx.register_operation(adapter)
        proof_edit_adapter, _ = install_lean_proof_edit_operation(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            resources.lean,
            runtime,
        )
        ctx.register_operation(proof_edit_adapter)

    def _bind_lean_declaration_adapters(
        self,
        declarations: LeanDeclarationService | None,
    ) -> None:
        if declarations is None:
            return
        ctx = self.context
        bound_queries = ctx.binder.bind(lean_declaration_query_operations(declarations))
        for adapter in bound_queries.adapters:
            ctx.register_operation(adapter)
