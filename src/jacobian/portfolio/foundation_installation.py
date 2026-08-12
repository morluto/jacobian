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

from jacobian.contracts.capabilities import (
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.installation.context import InstallationContext
from jacobian.polynomial_expression_capabilities import (
    install_polynomial_expression_checker,
)
from jacobian.portfolio.provider_resolution import ProviderRuntimePlan
from jacobian.runtime.services import CoreServices
from jacobian.sat_smt.cadical import install_cadical_capabilities
from jacobian.sat_smt.cvc5 import install_cvc5_capability
from jacobian.sat_smt.sat_capabilities import (
    SatCnfMaterializationAdapter,
    install_sat_assignment_checker,
    install_sat_unsat_proof_checker,
)
from jacobian.sat_smt.sat_lrat import install_sat_lrat_verifier
from jacobian.sat_smt.smt_capabilities import install_smt_unsat_proof_checker
from jacobian.sympy_polynomial_normalization import (
    install_sympy_polynomial_normalization_capability,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoundationInstaller:
    """Install foundational checkers and solver adapters."""

    context: InstallationContext

    def install(
        self,
        core: CoreServices,
        runtimes: ProviderRuntimePlan,
    ) -> None:
        """Install solver, linear, and normalization foundations."""

        ctx = self.context
        self.context.register_capability(SatCnfMaterializationAdapter(core.sat))
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
            self.context.register_capability(sat_assignment_adapter)

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
            self.context.register_capability(proof_adapter)

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
            self.context.register_capability(lrat_adapter)

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
            self.context.register_capability(smt_proof_adapter)

        self._install_polynomial_expression_capabilities(
            core,
            runtimes.sympy_polynomial_normalization,
        )
        self.install_solver_components(core, runtimes)

    def install_solver_components(
        self,
        core: CoreServices,
        runtimes: ProviderRuntimePlan,
    ) -> None:
        """Install the packaged cvc5 adapter and optional CaDiCaL adapter."""

        if runtimes.cadical.availability is CapabilityProviderAvailability.AVAILABLE:
            try:
                cadical_adapters = install_cadical_capabilities(
                    core.sat,
                    runtimes.cadical,
                )
            except OSError as exc:
                _LOGGER.warning("CaDiCaL SAT exploration is not installed: %s", exc)
            else:
                for adapter in cadical_adapters:
                    self.context.register_capability(adapter)

        self.context.register_capability(
            install_cvc5_capability(core.smt, runtimes.cvc5)
        )

    # ------------------------------------------------------------------
    # Private installation helpers
    # ------------------------------------------------------------------

    def _install_polynomial_expression_capabilities(
        self,
        core: CoreServices,
        runtime: CapabilityProviderRuntime,
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
            self.context.register_capability(verification_adapter)

        self.context.register_capability(
            install_sympy_polynomial_normalization_capability(
                core.polynomial_expressions,
                runtime,
            )
        )
