"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.analysis.approximation._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "approximation.lagrange.basis.compute",
        AdmissionDecision.KEEP,
        "exact Lagrange basis polynomials and barycentric weights over QQ",
    ),
    OperationAdmission(
        "approximation.lagrange.interpolate.compute",
        AdmissionDecision.NATIVE_ONLY,
        "duplicates polynomial.interpolation.newton_form.compute; the expanded "
        "polynomial representation stays a native jacobian.math value",
        native_symbol="jacobian.math.analysis.approximation.lagrange_interpolate",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
