"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.plane_algebraic_curves._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "algebraic_geometry.affine_plane_curve.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "algebraic_geometry.plane_curve.projective_closure.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "algebraic_geometry.conic.rational_parametrization.compute",
        AdmissionDecision.KEEP,
        "constructs one reusable birational conic chart with canonical rational "
        "functions, an inverse, and explicit exceptional loci that polynomial "
        "substitution alone cannot recover",
    ),
    OperationAdmission(
        "algebraic_geometry.projective_curve.affine_chart.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
