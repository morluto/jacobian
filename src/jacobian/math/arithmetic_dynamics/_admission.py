"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "arithmetic_dynamics.cycle.multiplier.compute",
        AdmissionDecision.KEEP,
        "exact multiplier spectrum of a rational map at a cycle",
    ),
    OperationAdmission(
        "arithmetic_dynamics.dynatomic_polynomial.compute",
        AdmissionDecision.KEEP,
        "exact dynatomic polynomial for a bounded iteration depth",
    ),
    OperationAdmission(
        "arithmetic_dynamics.finite_field.functional_graph.compute",
        AdmissionDecision.KEEP,
        "exact functional graph of a map over a finite field",
    ),
    OperationAdmission(
        "arithmetic_dynamics.map.iterate.compute",
        AdmissionDecision.KEEP,
        "exact rational map iterate over an explicit domain",
    ),
    OperationAdmission(
        "arithmetic_dynamics.point.orbit.compute",
        AdmissionDecision.KEEP,
        "exact orbit computation with typed periodicity",
    ),
)
