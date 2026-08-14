"""Explicit binding of solver and normalization catalog operations.

This module binds built-in adapters and checkers through one narrow
infrastructure context.
It performs no discovery, registration, ranking, or verification authorization
beyond what the individual installers do.

This phase never imports or accepts ``JacobianRuntime`` and never creates a
facade. Catalog compilation passes only the exact external observations used
by these operations.
"""

from __future__ import annotations

import logging

from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.polynomial_expression_operations import (
    install_polynomial_expression_checker,
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
from jacobian.sympy_polynomial_normalization import (
    bind_sympy_polynomial_normalization,
)

_LOGGER = logging.getLogger(__name__)


def bind_catalog_foundations(
    context: CatalogBuildContext,
    *,
    cadical: ProviderObservation,
    carcara: ProviderObservation,
    cvc5: ProviderObservation,
    drat_trim: ProviderObservation,
    sympy_normalization: ProviderObservation,
) -> None:
    """Bind the explicit solver and normalization operation families."""

    ctx = context
    context.register_operation(SatCnfMaterializationAdapter(ctx.sat))
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
        context.register_operation(sat_assignment_adapter)

    proof_adapter, _ = install_sat_unsat_proof_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.sat,
        ctx.verification,
        ctx.checkers,
        drat_trim,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if proof_adapter is not None:
        context.register_operation(proof_adapter)

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
        context.register_operation(lrat_adapter)

    smt_proof_adapter, _ = install_smt_unsat_proof_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        ctx.smt,
        ctx.verification,
        ctx.checkers,
        carcara,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if smt_proof_adapter is not None:
        context.register_operation(smt_proof_adapter)

    _bind_polynomial_expression_operations(context, sympy_normalization)
    _bind_solver_components(context, cadical=cadical, cvc5=cvc5)


def _bind_solver_components(
    context: CatalogBuildContext,
    *,
    cadical: ProviderObservation,
    cvc5: ProviderObservation,
) -> None:
    """Bind the packaged cvc5 adapter and optional CaDiCaL adapter."""

    if cadical.availability is ProviderAvailability.AVAILABLE:
        try:
            cadical_adapters = install_cadical_operations(context.sat, cadical)
        except OSError as exc:
            _LOGGER.warning("CaDiCaL SAT exploration is not installed: %s", exc)
        else:
            for adapter in cadical_adapters:
                context.register_operation(adapter)

    context.register_operation(bind_cvc5_operation(context.smt, cvc5))


def _bind_polynomial_expression_operations(
    context: CatalogBuildContext,
    runtime: ProviderObservation,
) -> None:
    ctx = context
    verification_adapter, _ = install_polynomial_expression_checker(
        ctx.store,
        ctx.schemas,
        ctx.artifacts,
        context.polynomial_expressions,
        ctx.verification,
        ctx.checkers,
        authorize_checker=ctx.authorize_bundled_checkers,
    )
    if verification_adapter is not None:
        context.register_operation(verification_adapter)

    context.register_operation(
        bind_sympy_polynomial_normalization(
            context.polynomial_expressions,
            runtime,
        )
    )


__all__ = ["bind_catalog_foundations"]
