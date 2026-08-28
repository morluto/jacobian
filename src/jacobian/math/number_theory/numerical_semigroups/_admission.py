"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.numerical_semigroups._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.numerical_semigroup.betti_elements.compute",
        AdmissionDecision.KEEP,
        "exact complete Betti enumeration rather than a capped heuristic search",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.catenary_degree.compute",
        AdmissionDecision.KEEP,
        "complete global catenary-degree invariant on the exact Betti basis",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.delta_set.compute",
        AdmissionDecision.KEEP,
        "complete global delta-set invariant on the exact Betti basis",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.elasticity.compute",
        AdmissionDecision.KEEP,
        "exact per-element elasticity as the max/min factorization-length ratio",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.elasticity.global_compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap generator-ratio projection already available via summary and native elasticity",
        native_symbol="jacobian.math.number_theory.numerical_semigroups.elasticity",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.factorization_graph.compute",
        AdmissionDecision.KEEP,
        "reusable factorization graph and component construction",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.factorizations.compute",
        AdmissionDecision.KEEP,
        "complete bounded factorization family for one element",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.membership.compute",
        AdmissionDecision.KEEP,
        "exact bounded semigroup-membership predicate with material "
        "reliability leverage and a catalog-discoverable request envelope",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        AdmissionDecision.KEEP,
        "minimal presentation constructed from the exact Betti basis",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.presentation_binomials.compute",
        AdmissionDecision.KEEP,
        "unit binomial coefficients of the exact minimal presentation",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.summary.compute",
        AdmissionDecision.KEEP,
        "one complete exact finite gap profile with its mutually determined "
        "canonical invariants on the normalized increasing generator axis",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
