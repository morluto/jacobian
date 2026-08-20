"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.sequences._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "sequence.compute.distinct_count",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.first_differences",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.frequencies",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.gcd",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.lcm",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.maximum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.mean",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.median",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.minimum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_gcds",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_lcms",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_maxima",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_minima",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_products",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.prefix_sums",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.product",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.range",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.second_differences",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.sum",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.compute.zero_indices",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.arithmetic",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.geometric",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.nondecreasing",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.decide.strictly_increasing",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.parities",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.reverse",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.signs",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.sort",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
    OperationAdmission(
        "sequence.transform.sorted_unique",
        AdmissionDecision.DROP,
        "ordinary deterministic finite-sequence projection better authored directly in Python",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
