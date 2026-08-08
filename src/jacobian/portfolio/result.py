"""Typed installation results and diagnostics owned by the portfolio.

These types are self-contained: they do not depend on a shared installation
result module and they carry no verification authority. Diagnostics record
non-conclusive installation observations only; a diagnostic never promotes a
skipped bundle into an installed one.

The only per-bundle omission the portfolio records is a declared unavailable
provider. Every other installation failure propagates to the caller so the
enclosing runtime transaction rolls back atomically, so there is no
"install-failed" status here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from jacobian.conjecture_ingestion import ConjectureIngestionInstallation
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.lean import LeanEnvironment
from jacobian.exact_domain_checkers import ExactDomainCheckerInstallation
from jacobian.finite_coverage import FiniteCoverageInstallation
from jacobian.finite_partition import FinitePartitionInstallation
from jacobian.graphs.coloring import GraphColoringInstallation
from jacobian.graphs.composition import GraphCompositionInstallation
from jacobian.graphs.installation import GraphInstallation
from jacobian.graphs.isomorphism import GraphIsomorphismInstallation
from jacobian.graphs.shrinking import GraphShrinkingInstallation
from jacobian.lean_frontend.declarations import LeanDeclarationService
from jacobian.lean_frontend.exploration import LeanExplorationInstallation
from jacobian.lean_frontend.proof_axioms import LeanProofAxiomsInstallation
from jacobian.lean_frontend.proof_edit import LeanProofEditInstallation
from jacobian.lean_frontend.service import LeanService
from jacobian.lean_frontend.statement import LeanStatementInstallation
from jacobian.matrices.linear_capabilities import (
    LinearRationalInconsistencyCheckerInstallation,
    LinearRationalSolutionCheckerInstallation,
)
from jacobian.matrices.normal_form import (
    MatrixNormalFormCheckerInstallation,
)
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.polynomial_expression_capabilities import (
    PolynomialExpressionCheckerInstallation,
)
from jacobian.polynomial_interval_capabilities import PolynomialIntervalInstallation
from jacobian.polynomial_positivity_capabilities import (
    PolynomialPositivityInstallation,
)
from jacobian.polynomial_system_capabilities import PolynomialSystemInstallation
from jacobian.polynomials import PolynomialInstallation
from jacobian.references import (
    LeanCheckerInstallation,
    PolytopeCheckerInstallation,
    ReferenceInstallation,
)
from jacobian.sat_smt.sat_capabilities import (
    SatAssignmentCheckerInstallation,
    SatUnsatProofCheckerInstallation,
)
from jacobian.sat_smt.sat_lrat import SatLratInstallation
from jacobian.sat_smt.smt_capabilities import SmtUnsatProofCheckerInstallation
from jacobian.universal_algebra_capabilities import UniversalAlgebraInstallation

# Diagnostic codes follows the same convention as CapabilityDiagnostic codes.
PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
_DIAGNOSTIC_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class BundleInstallationStatus(StrEnum):
    """The lifecycle of one domain bundle within a portfolio installation."""

    INSTALLED = "INSTALLED"
    SKIPPED_PROVIDER_UNAVAILABLE = "SKIPPED_PROVIDER_UNAVAILABLE"
    SKIPPED_DEPENDENCY_UNAVAILABLE = "SKIPPED_DEPENDENCY_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostic:
    """One inspectable, non-conclusive portfolio installation observation.

    Diagnostics are fail-closed: they describe why a bundle was skipped. They
    never assert a mathematical conclusion and never authorize a checker.
    """

    code: str
    component_id: str
    stage: str
    message: str

    def __post_init__(self) -> None:
        if not _DIAGNOSTIC_CODE_PATTERN.fullmatch(self.code):
            raise ValueError("portfolio diagnostic code has an invalid format")


@dataclass(frozen=True, slots=True)
class BundleInstallation:
    """Per-bundle outcome of installing one domain bundle.

    ``capability_ids`` is populated from the declared bundle even when the
    bundle was skipped, so callers can see exactly which capabilities are
    absent from the installed portfolio.
    """

    domain_id: str
    status: BundleInstallationStatus
    capability_ids: tuple[str, ...]
    installed: InstalledDomainBundle | None
    diagnostic: PortfolioDiagnostic | None

    def __post_init__(self) -> None:
        installed = self.status is BundleInstallationStatus.INSTALLED
        if installed != (self.installed is not None):
            raise ValueError("bundle installation status and value disagree")
        if installed == (self.diagnostic is not None):
            raise ValueError("bundle installation status and diagnostic disagree")


@dataclass(frozen=True, slots=True)
class PortfolioInstallationResult:
    """Typed result of installing a :class:`PortfolioPlan` (domain bundles only).

    ``installed`` is the mapping consumed by the runtime (domain_id to the
    installed bundle). ``outcomes`` preserves installation order and records a
    status for every declared bundle, including skipped ones. ``diagnostics``
    is the subset of outcomes that prevented installation.
    """

    installed: Mapping[str, InstalledDomainBundle]
    diagnostics: tuple[PortfolioDiagnostic, ...]
    outcomes: tuple[BundleInstallation, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            outcome.diagnostic
            for outcome in self.outcomes
            if outcome.diagnostic is not None
        )
        if self.diagnostics != expected:
            raise ValueError("portfolio diagnostics do not match bundle outcomes")
        object.__setattr__(
            self,
            "installed",
            MappingProxyType(dict(self.installed)),
        )

    @property
    def is_complete(self) -> bool:
        """True when every declared bundle was installed without diagnostics."""

        return not self.diagnostics

    @property
    def installed_domain_ids(self) -> tuple[str, ...]:
        """Domain IDs that were installed, in installation order."""

        return tuple(
            outcome.domain_id
            for outcome in self.outcomes
            if outcome.status is BundleInstallationStatus.INSTALLED
        )

    @property
    def skipped_domain_ids(self) -> tuple[str, ...]:
        """Domain IDs that were skipped, in declaration order."""

        return tuple(
            outcome.domain_id
            for outcome in self.outcomes
            if outcome.status is not BundleInstallationStatus.INSTALLED
        )

    def outcome_for(self, domain_id: str) -> BundleInstallation | None:
        """Return the per-bundle outcome for ``domain_id``, or ``None``."""

        for outcome in self.outcomes:
            if outcome.domain_id == domain_id:
                return outcome
        return None

    def diagnostic_for(self, domain_id: str) -> PortfolioDiagnostic | None:
        """Return the diagnostic that skipped ``domain_id``, or ``None``."""

        outcome = self.outcome_for(domain_id)
        return None if outcome is None else outcome.diagnostic


@dataclass(slots=True)
class PortfolioInstallation:
    """Typed result of installing the complete built-in portfolio.

    Carries every installed component result and provider-runtime metadata that
    ``JacobianRuntime._install_capability_portfolio`` previously assigned
    dynamically on the runtime instance. Fields default to ``None`` or empty so
    the installer can populate them incrementally; a ``None`` field means the
    component was not installed (typically because an optional provider was
    declared unavailable).
    """

    # --- SAT / SMT checkers ---
    sat_assignment_checker: SatAssignmentCheckerInstallation | None = None
    sat_unsat_proof_checker: SatUnsatProofCheckerInstallation | None = None
    sat_lrat: SatLratInstallation | None = None
    smt_unsat_proof_checker: SmtUnsatProofCheckerInstallation | None = None

    # --- Provider runtimes ---
    drat_trim_runtime: CapabilityProviderRuntime | None = None
    carcara_runtime: CapabilityProviderRuntime | None = None
    cadical_runtime: CapabilityProviderRuntime | None = None
    cvc5_runtime: CapabilityProviderRuntime | None = None
    python_flint_runtime: CapabilityProviderRuntime | None = None
    python_flint_hnf_runtime: CapabilityProviderRuntime | None = None
    sympy_polynomial_normalization_runtime: CapabilityProviderRuntime | None = None
    lean_runtime: CapabilityProviderRuntime | None = None

    # --- Linear checkers ---
    linear_solution_checker: LinearRationalSolutionCheckerInstallation | None = None
    linear_inconsistency_checker: (
        LinearRationalInconsistencyCheckerInstallation | None
    ) = None

    # --- Matrix ---
    matrix_normal_form_checker: MatrixNormalFormCheckerInstallation | None = None

    # --- Polynomial ---
    polynomial: PolynomialInstallation | None = None
    polynomial_expression_checker: PolynomialExpressionCheckerInstallation | None = None
    polynomial_system: PolynomialSystemInstallation | None = None
    polynomial_interval: PolynomialIntervalInstallation | None = None
    polynomial_positivity: PolynomialPositivityInstallation | None = None

    # --- Graph ---
    graph: GraphInstallation | None = None
    graph_shrinking: GraphShrinkingInstallation | None = None
    graph_coloring: GraphColoringInstallation | None = None
    graph_isomorphism: GraphIsomorphismInstallation | None = None
    graph_composition: GraphCompositionInstallation | None = None

    # --- Finite ---
    finite_partition: FinitePartitionInstallation | None = None
    finite_coverage: FiniteCoverageInstallation | None = None

    # --- Datasets ---
    conjecture_ingestion: ConjectureIngestionInstallation | None = None

    # --- Domain bundles ---
    domain_bundles: dict[str, InstalledDomainBundle] = field(default_factory=dict)
    portfolio_diagnostics: tuple[PortfolioDiagnostic, ...] = ()
    portfolio_outcomes: tuple[BundleInstallation, ...] = ()

    # --- Verification ---
    exact_domain_checkers: ExactDomainCheckerInstallation | None = None

    # --- Universal algebra ---
    universal_algebra: UniversalAlgebraInstallation | None = None

    # --- Lean ---
    lean_statement: LeanStatementInstallation | None = None
    lean_statement_runtime: CapabilityProviderRuntime | None = None
    lean: LeanService | None = None
    lean_declarations: LeanDeclarationService | None = None
    lean_exploration: LeanExplorationInstallation | None = None
    lean_proof_edit: LeanProofEditInstallation | None = None
    lean_proof_axioms: LeanProofAxiomsInstallation | None = None

    # --- References ---
    references: dict[str, ReferenceInstallation] = field(default_factory=dict)
    polytope_checkers: PolytopeCheckerInstallation | None = None
    lean_checkers: dict[LeanEnvironment, LeanCheckerInstallation] = field(
        default_factory=dict
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Release portfolio-owned Lean sessions, processes, and warm-up work."""

        if self._closed:
            return
        failures: list[Exception] = []
        resources = (
            self.lean_declarations,
            self.lean_exploration.repl if self.lean_exploration is not None else None,
            self.lean,
        )
        for resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("portfolio resources failed to close", failures)
        self._closed = True

    def installed_bundle(self, domain_id: str) -> InstalledDomainBundle | None:
        """Return one installed mathematical domain bundle, if available."""

        return self.domain_bundles.get(domain_id)

    def outcome_for(self, domain_id: str) -> BundleInstallation | None:
        """Return the ordered installation outcome for one mathematical domain."""

        for outcome in self.portfolio_outcomes:
            if outcome.domain_id == domain_id:
                return outcome
        return None

    @property
    def installed_domain_ids(self) -> tuple[str, ...]:
        return tuple(
            outcome.domain_id
            for outcome in self.portfolio_outcomes
            if outcome.status is BundleInstallationStatus.INSTALLED
        )

    @property
    def skipped_domain_ids(self) -> tuple[str, ...]:
        return tuple(
            outcome.domain_id
            for outcome in self.portfolio_outcomes
            if outcome.status is not BundleInstallationStatus.INSTALLED
        )
