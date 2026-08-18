"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

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
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
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
        "one complete exact finite gap profile with its mutually determined canonical invariants",
    ),
)
