"""Owner-local admission decisions for built-in lattice-polytope operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.geometry.polytopes.lattice._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polytope.lattice_points.enumerate",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value with material "
        "computational and reliability leverage: the complete list of "
        "lattice points inside a bounded rational polytope is a composable "
        "building block for Ehrhart theory and polyhedral geometry",
    ),
    OperationAdmission(
        "polytope.lattice_points.count",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value with material "
        "computational and reliability leverage: the lattice-point count "
        "of a bounded rational polytope is a composable building block for "
        "Ehrhart theory and polyhedral geometry",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
