"""Installation of solver, linear, and normalization foundations.

This phase installs built-in adapters and checkers through one narrow
infrastructure context plus the core service graph.
It performs no discovery, registration, ranking, or verification authorization
beyond what the individual installers do.

This phase never imports or accepts ``JacobianRuntime`` and never creates a
facade. It consumes the explicit provider-runtime plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.polynomial_expression_operations import (
    install_polynomial_expression_checker,
)
from jacobian.portfolio.context import PortfolioContext
from jacobian.portfolio.provider_resolution import ProviderRuntimePlan
from jacobian.runtime.services import CoreServices
from jacobian.sat_smt.cadical import install_cadical_capabilities
from jacobian.sat_smt.cvc5 import install_cvc5_operation
from jacobian.sat_smt.sat_lrat import install_sat_lrat_verifier
from jacobian.sat_smt.sat_operations import (
    SatCnfMaterializationAdapter,
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)
from jacobian.sat_smt.smt_operations import install_smt_unsat_proof_checker
from jacobian.sympy_polynomial_normalization import (
    install_sympy_polynomial_normalization_operation,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoundationBinder:
    """Bind foundational checkers and solver operations."""

    context: PortfolioContext

    def bind(
        self,
        core: CoreServices,
        runtimes: ProviderRuntimePlan,
    ) -> None:
        """Bind solver, linear, and normalization foundations."""

        ctx = self.context
        self.context.register_operation(SatCnfMaterializationAdapter(core.sat))
        sat_assignment_adapter, _ = install_sat_assignment_checker(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.sat,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if sat_assignment_adapter is not None:
            self.context.register_operation(sat_assignment_adapter)

        proof_adapter, _ = install_sat_unsat_proof_checker(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.sat,
            ctx.verification,
            ctx.checkers,
            runtimes.drat_trim,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if proof_adapter is not None:
            self.context.register_operation(proof_adapter)

        lrat_adapter, _ = install_sat_lrat_verifier(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.sat,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if lrat_adapter is not None:
            self.context.register_operation(lrat_adapter)

        smt_proof_adapter, _ = install_smt_unsat_proof_checker(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.smt,
            ctx.verification,
            ctx.checkers,
            runtimes.carcara,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if smt_proof_adapter is not None:
            self.context.register_operation(smt_proof_adapter)

        self._bind_polynomial_expression_operations(
            core,
            runtimes.sympy_polynomial_normalization,
        )
        self.bind_solver_components(core, runtimes)

    def bind_solver_components(
        self,
        core: CoreServices,
        runtimes: ProviderRuntimePlan,
    ) -> None:
        """Bind the packaged cvc5 adapter and optional CaDiCaL adapter."""

        if runtimes.cadical.availability is ProviderAvailability.AVAILABLE:
            try:
                cadical_adapters = install_cadical_capabilities(
                    core.sat,
                    runtimes.cadical,
                )
            except OSError as exc:
                _LOGGER.warning("CaDiCaL SAT exploration is not installed: %s", exc)
            else:
                for adapter in cadical_adapters:
                    self.context.register_operation(adapter)

        self.context.register_operation(install_cvc5_operation(core.smt, runtimes.cvc5))

    # ------------------------------------------------------------------
    # Private binding helpers
    # ------------------------------------------------------------------

    def _bind_polynomial_expression_operations(
        self,
        core: CoreServices,
        runtime: ProviderObservation,
    ) -> None:
        ctx = self.context
        verification_adapter, _ = install_polynomial_expression_checker(
            ctx.store,
            ctx.schemas,
            ctx.artifacts,
            core.polynomial_expressions,
            ctx.verification,
            ctx.checkers,
            authorize_checker=ctx.authorizes_bundled_checkers,
        )
        if verification_adapter is not None:
            self.context.register_operation(verification_adapter)

        self.context.register_operation(
            install_sympy_polynomial_normalization_operation(
                core.polynomial_expressions,
                runtime,
            )
        )
