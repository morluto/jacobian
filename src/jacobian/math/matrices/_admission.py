"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.matrices._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "matrix.adjugate.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.adjugate",
    ),
    OperationAdmission(
        "matrix.characteristic_polynomial.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.determinant.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.kronecker_product.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.kronecker_product",
    ),
    OperationAdmission(
        "matrix.multiply.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.multiply",
    ),
    OperationAdmission(
        "matrix.normal_form.rref.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.normal_form.smith.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.nullspace.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.partial_trace.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.permanent.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.rank.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "matrix.rational_linear_system.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "matrix.trace.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.matrices.trace",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
