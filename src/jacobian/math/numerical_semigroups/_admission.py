"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.numerical_semigroups._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.numerical_semigroup.betti_elements.compute",
        AdmissionDecision.KEEP,
        "exact complete Betti enumeration replacing the capped heuristic search after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.catenary_degree.compute",
        AdmissionDecision.KEEP,
        "complete global catenary-degree invariant rebuilt on the repaired Betti basis after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.delta_set.compute",
        AdmissionDecision.KEEP,
        "complete global delta-set invariant rebuilt on the repaired Betti basis after the #1977 contract repair",
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
        native_symbol="jacobian.math.numerical_semigroups.elasticity",
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
        "reliability leverage; decision renewed when the request schema began "
        "publishing its admission envelope - positive gcd-one generators at "
        "most 500 (at most 20) and a tested value at most 10000 - through "
        "catalog-discoverable field descriptions",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.minimal_presentation.compute",
        AdmissionDecision.KEEP,
        "minimal presentation rebuilt on the exact Betti basis after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.presentation_binomials.compute",
        AdmissionDecision.KEEP,
        "unit binomial coefficients of the repaired minimal presentation after the #1977 contract repair",
    ),
    OperationAdmission(
        "number_theory.numerical_semigroup.summary.compute",
        AdmissionDecision.KEEP,
        "one complete exact finite gap profile with its mutually determined "
        "canonical invariants; decision renewed when the request and result "
        "schemas began publishing the admission envelope - positive gcd-one "
        "generators at most 500 (at most 20) normalized to the increasing "
        "minimal generator axis - through catalog-discoverable field "
        "descriptions",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
