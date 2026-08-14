"""Lazy binding for the selected SAT/SMT operation family."""

from __future__ import annotations

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


__all__ = [
    "SELECTED_SAT_SMT_OPERATION_IDS",
    "bind_selected_sat_smt_operation",
]
