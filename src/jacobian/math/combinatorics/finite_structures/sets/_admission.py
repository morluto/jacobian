"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.combinatorics.finite_structures.sets._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "finite_set.compute.difference",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.intersection",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.intersection_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.left_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.symmetric_difference",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.union",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.compute.union_cardinality",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.disjoint",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.exact_cover",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.proper_subset",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
    OperationAdmission(
        "finite_set.decide.subset",
        AdmissionDecision.DROP,
        "ordinary deterministic set projection better authored directly in Python",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
