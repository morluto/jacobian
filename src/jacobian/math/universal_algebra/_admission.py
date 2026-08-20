"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.universal_algebra._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "universal_algebra.term.evaluate.compute",
        AdmissionDecision.KEEP,
        "exact bottom-up term evaluation over a finite algebra with complete operation tables",
    ),
    OperationAdmission(
        "universal_algebra.equation.profile.compute",
        AdmissionDecision.KEEP,
        "exact equation profile with complete assignment enumeration and first counterassignment",
    ),
    OperationAdmission(
        "universal_algebra.subalgebra.generated.compute",
        AdmissionDecision.KEEP,
        "exact least subalgebra by finite closure under all basic operations and nullary constants",
    ),
    OperationAdmission(
        "universal_algebra.congruence.check.compute",
        AdmissionDecision.KEEP,
        "exact compatibility check of a carrier partition against all basic operations",
    ),
    OperationAdmission(
        "universal_algebra.quotient.compute",
        AdmissionDecision.KEEP,
        "exact quotient algebra induced by a congruence with block-wise operations",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
