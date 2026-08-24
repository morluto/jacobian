"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.symbolic_dynamics._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "symbolic_dynamics.artin_mazur_zeta.compute",
        AdmissionDecision.KEEP,
        "exact rational dynamical invariant with periodic-point replay",
    ),
    OperationAdmission(
        "symbolic_dynamics.block_language.compute",
        AdmissionDecision.KEEP,
        "complete occurring-block language of a bounded shift presentation",
    ),
    OperationAdmission(
        "symbolic_dynamics.finite_type_shift.construct",
        AdmissionDecision.KEEP,
        "exact finite-type shift presentation from a bounded edge label set",
    ),
    OperationAdmission(
        "symbolic_dynamics.higher_block.compute",
        AdmissionDecision.KEEP,
        "exact higher-block presentation of a bounded shift",
    ),
    OperationAdmission(
        "symbolic_dynamics.periodic_point_profile.compute",
        AdmissionDecision.KEEP,
        "complete periodic point profile of a bounded shift",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
