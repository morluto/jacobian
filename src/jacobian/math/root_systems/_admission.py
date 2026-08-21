"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.root_systems._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "root_system.positive_roots.compute",
        AdmissionDecision.KEEP,
        "complete positive roots and componentwise finite-root invariants",
    ),
    OperationAdmission(
        "root_system.simple_reflection.compute",
        AdmissionDecision.KEEP,
        "exact simple reflection s_i on a root lattice vector",
    ),
    OperationAdmission(
        "root_system.weyl_group_data.compute",
        AdmissionDecision.KEEP,
        "exact Weyl group order, longest element, Coxeter number from Cartan matrix",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
