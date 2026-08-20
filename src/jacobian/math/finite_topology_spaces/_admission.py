"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_topology_spaces._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "topology.finite.interior.compute",
        AdmissionDecision.KEEP,
        "exact interior via minimal open neighbourhood containment in an Alexandrov space",
    ),
    OperationAdmission(
        "topology.finite.closure.compute",
        AdmissionDecision.KEEP,
        "exact closure via specialization preorder up-set",
    ),
    OperationAdmission(
        "topology.finite.boundary.compute",
        AdmissionDecision.KEEP,
        "exact boundary as closure minus interior",
    ),
    OperationAdmission(
        "topology.finite.kolmogorov_quotient.compute",
        AdmissionDecision.KEEP,
        "exact T0 quotient identifying points with the same minimal open neighbourhood",
    ),
    OperationAdmission(
        "topology.finite.continuity_check.compute",
        AdmissionDecision.KEEP,
        "exact continuity check via specialization preorder monotonicity",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
