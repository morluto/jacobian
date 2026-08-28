"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.series._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "formal_series.rational.add.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.add",
    ),
    OperationAdmission(
        "formal_series.rational.compose.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "formal_series.rational.derivative.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.derivative",
    ),
    OperationAdmission(
        "formal_series.rational.divide.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.from_polynomial.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.from_polynomial",
    ),
    OperationAdmission(
        "formal_series.rational.identity.check",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.identity_check",
    ),
    OperationAdmission(
        "formal_series.rational.integral_zero_constant.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.integral_zero_constant",
    ),
    OperationAdmission(
        "formal_series.rational.inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.multiply.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.power.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.reversion.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "formal_series.rational.scalar_multiply.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.scalar_multiply",
    ),
    OperationAdmission(
        "formal_series.rational.subtract.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.subtract",
    ),
    OperationAdmission(
        "formal_series.rational.to_polynomial.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.to_polynomial",
    ),
    OperationAdmission(
        "formal_series.rational.truncate.compute",
        AdmissionDecision.NATIVE_ONLY,
        "cheap structural projection of the supplied truncated-series value",
        native_symbol="jacobian.math.polynomials.series.truncate",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
