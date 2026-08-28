"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.elliptic_curves._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "number_theory.elliptic_curve.short_weierstrass.discriminant.compute",
        AdmissionDecision.KEEP,
        "exact elliptic curve discriminant and nonsingularity predicate",
    ),
    OperationAdmission(
        "number_theory.elliptic_curve.short_weierstrass.point_on_curve.decide",
        AdmissionDecision.KEEP,
        "exact point-on-curve predicate for short Weierstrass curves",
    ),
    OperationAdmission(
        "number_theory.elliptic_curve.short_weierstrass.point_addition.compute",
        AdmissionDecision.KEEP,
        "exact chord-and-tangent group law addition on short Weierstrass curves",
    ),
    OperationAdmission(
        "number_theory.elliptic_curve.short_weierstrass.scalar_multiply.compute",
        AdmissionDecision.KEEP,
        "exact scalar multiplication via double-and-add on short Weierstrass curves",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
