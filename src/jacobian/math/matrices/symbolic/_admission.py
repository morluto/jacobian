"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matrices.symbolic._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.symbolic.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.determinant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.eigenvalues.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.symbolic.linear_system.solve",
        AdmissionDecision.KEEP,
        "exact classification and generic solution data for one bounded linear system over QQ(t_1, ..., t_n)",
    ),
    OperationAdmission(
        "matrix.symbolic.multiply.compute",
        AdmissionDecision.KEEP,
        "exact bounded symbolic matrix product over one canonical ordered rational-function field",
    ),
    OperationAdmission(
        "matrix.symbolic.rank.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
