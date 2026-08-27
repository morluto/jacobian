"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_stochastic_processes._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "probability.finite_sigma_algebra.from_observation.compute",
        AdmissionDecision.KEEP,
        "exact sigma algebra construction from an observation map with equal-value fibers",
    ),
    OperationAdmission(
        "probability.finite_sigma_algebra.join.compute",
        AdmissionDecision.KEEP,
        "exact join of two finite sigma algebras as the finest common refinement",
    ),
    OperationAdmission(
        "probability.conditional_expectation.finite.compute",
        AdmissionDecision.KEEP,
        "exact block-constant conditional expectation with probability-weighted block averages",
    ),
    OperationAdmission(
        "probability.filtration.natural.compute",
        AdmissionDecision.KEEP,
        "exact natural filtration F_t = sigma(Y_0, ..., Y_t) with monotone refinement",
    ),
    OperationAdmission(
        "probability.process.doob_martingale.compute",
        AdmissionDecision.KEEP,
        "exact Doob martingale M_t = E[payoff | F_t] with rational-valued conditional expectations",
    ),
    OperationAdmission(
        "probability.poisson_binomial.distribution.compute",
        AdmissionDecision.KEEP,
        "distinct exact Poisson-binomial count distribution that a loop of scalar Bernoulli convolution calls cannot establish as one source-bound value",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
