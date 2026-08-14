"""Lazy binding for the selected SAT/SMT operation family."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.registry import CheckerRegistry
from jacobian.sat_smt.sat import SatArtifactService
from jacobian.sat_smt.smt import SmtArtifactService
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.catalog_build_context import CatalogBuildContext

_LOGGER = logging.getLogger(__name__)

SELECTED_SAT_SMT_OPERATION_IDS = frozenset(
    {
        "sat.cnf.materialize",
        "sat.model.verify",
        "sat.unsat_proof.verify",
        "sat.lrat.verify",
        "smt.unsat_proof.verify",
        "sat.model.find",
        "sat.unsat_proof.find",
        "smt.unsat_proof.find",
    }
)


def bind_selected_sat_smt_operation(
    operation_id: str,
    descriptor: OperationDescriptor,
    *,
    binder: OperationBinder,
    verification: VerificationService,
    checkers: CheckerRegistry,
    catalog: OperationCatalog,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    sat: SatArtifactService,
    smt: SmtArtifactService,
) -> SelectedOperationBinding | None:
    """Bind one selected SAT/SMT operation from runtime-owned services."""

    if operation_id not in SELECTED_SAT_SMT_OPERATION_IDS:
        return None
    if operation_id in {"sat.model.find", "sat.unsat_proof.find"}:
        from jacobian.providers.external_solver_runtime import (
            cadical_provider_runtime,
        )
        from jacobian.sat_smt.cadical import install_cadical_operations

        adapter = next(
            adapter
            for adapter in install_cadical_operations(
                sat,
                cadical_provider_runtime(),
            )
            if adapter.descriptor.operation_id == operation_id
        )
        return SelectedOperationBinding(adapter)
    if operation_id == "smt.unsat_proof.find":
        from jacobian.providers.external_solver_runtime import (
            cvc5_provider_runtime,
        )
        from jacobian.sat_smt.cvc5 import bind_cvc5_operation

        return SelectedOperationBinding(
            bind_cvc5_operation(smt, cvc5_provider_runtime())
        )
    if operation_id == "sat.cnf.materialize":
        from jacobian.sat_smt.sat_operations import SatCnfMaterializationAdapter

        return SelectedOperationBinding(SatCnfMaterializationAdapter(sat))
    if operation_id in {"sat.model.verify", "sat.unsat_proof.verify"}:
        from jacobian.sat_smt.sat_operations import (
            bind_selected_sat_verification,
        )

        verification_adapter = bind_selected_sat_verification(
            operation_id,
            descriptor,
            store,
            schemas,
            binder.artifacts,
            sat,
            verification,
            checkers,
            catalog,
        )
        return (
            None
            if verification_adapter is None
            else SelectedOperationBinding(verification_adapter)
        )
    if operation_id == "sat.lrat.verify":
        from jacobian.sat_smt.sat_lrat import bind_selected_sat_lrat_verifier

        return SelectedOperationBinding(
            bind_selected_sat_lrat_verifier(
                store,
                schemas,
                binder.artifacts,
                sat,
                verification,
                checkers,
                catalog,
            )
        )
    from jacobian.sat_smt.smt_operations import (
        bind_selected_smt_unsat_proof_checker,
    )

    return SelectedOperationBinding(
        bind_selected_smt_unsat_proof_checker(
            descriptor,
            store,
            schemas,
            binder.artifacts,
            smt,
            verification,
            checkers,
            catalog,
        )
    )


def install_selected_sat_smt_catalog(
    context: CatalogBuildContext,
    *,
    polytope: object | None = None,
    resources: object | None = None,
) -> None:
    """Compile CNF, SAT/SMT checkers, CaDiCaL, cvc5, and LRAT operations."""

    del polytope, resources
    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.providers.external_solver_runtime import (
        cadical_provider_runtime,
        carcara_provider_runtime,
        cvc5_provider_runtime,
        drat_trim_provider_runtime,
    )
    from jacobian.sat_smt.cadical import install_cadical_operations
    from jacobian.sat_smt.cvc5 import bind_cvc5_operation
    from jacobian.sat_smt.sat_lrat import install_sat_lrat_verifier
    from jacobian.sat_smt.sat_operations import (
        SatCnfMaterializationAdapter,
        install_sat_assignment_checker,
        install_sat_unsat_proof_checker,
    )
    from jacobian.sat_smt.smt_operations import install_smt_unsat_proof_checker

    ctx = context
    ctx.register_operation(SatCnfMaterializationAdapter(ctx.sat))
    sat_assignment_adapter, _ = install_sat_assignment_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.sat,
        ctx.verification,
        ctx.checkers,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if sat_assignment_adapter is not None:
        ctx.register_operation(sat_assignment_adapter)

    proof_adapter, _ = install_sat_unsat_proof_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.sat,
        ctx.verification,
        ctx.checkers,
        drat_trim_provider_runtime(),
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if proof_adapter is not None:
        ctx.register_operation(proof_adapter)

    lrat_adapter, _ = install_sat_lrat_verifier(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.sat,
        ctx.verification,
        ctx.checkers,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if lrat_adapter is not None:
        ctx.register_operation(lrat_adapter)

    smt_proof_adapter, _ = install_smt_unsat_proof_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.smt,
        ctx.verification,
        ctx.checkers,
        carcara_provider_runtime(),
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if smt_proof_adapter is not None:
        ctx.register_operation(smt_proof_adapter)

    cadical = cadical_provider_runtime()
    if cadical.availability is ProviderAvailability.AVAILABLE:
        try:
            cadical_adapters = install_cadical_operations(ctx.sat, cadical)
        except OSError as exc:
            _LOGGER.warning("CaDiCaL SAT exploration is not installed: %s", exc)
        else:
            for adapter in cadical_adapters:
                ctx.register_operation(adapter)

    ctx.register_operation(bind_cvc5_operation(ctx.smt, cvc5_provider_runtime()))


__all__ = [
    "SELECTED_SAT_SMT_OPERATION_IDS",
    "bind_selected_sat_smt_operation",
    "install_selected_sat_smt_catalog",
]
