"""Owner-local admission decisions for polynomial support geometry."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.support_geometry._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.support.compute",
        AdmissionDecision.KEEP,
        "exact exponent support of a nonzero sparse polynomial",
    ),
    OperationAdmission(
        "polynomial.newton_polytope.compute",
        AdmissionDecision.KEEP,
        "exact Newton polytope: convex hull of support exponents",
    ),
    OperationAdmission(
        "polynomial.weight_profile.compute",
        AdmissionDecision.KEEP,
        "exact weight profile of a polynomial support under an integer weight vector",
    ),
    OperationAdmission(
        "polynomial.initial_form.compute",
        AdmissionDecision.KEEP,
        "exact initial form: sum of minimum-weight terms",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
