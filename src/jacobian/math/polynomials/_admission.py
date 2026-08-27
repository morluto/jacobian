"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.compute.discriminant",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.resultant",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.compute.square_free_decomposition",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "polynomial.factor.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.integer.compute.compose",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.content",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.evaluate",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.gcd",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.integer.compute.primitive_part",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.integer.compute.shift",
        AdmissionDecision.DROP,
        "ordinary polynomial projection or composition better expressed through the native SymPy-valued API",
    ),
    OperationAdmission(
        "polynomial.jacobian_syzygy.coefficients.compute",
        AdmissionDecision.KEEP,
        "dimension-parametric exact sparse ledger with separately bounded SymPy rank replay and material certificate leverage; labelled projective-line factors remain the compatible three-variable input",
    ),
    OperationAdmission(
        "polynomial.jacobian_syzygy.minimum_degree.compute",
        AdmissionDecision.KEEP,
        "dimension-parametric exact first-kernel search with separately bounded SymPy rank replay and material computational leverage; labelled projective-line factors remain the compatible three-variable input",
    ),
    OperationAdmission(
        "polynomial.rational.compute.derivative",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.derivative",
    ),
    OperationAdmission(
        "polynomial.rational.compute.evaluate",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.evaluate",
    ),
    OperationAdmission(
        "polynomial.rational.compute.integral",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.integral",
    ),
    OperationAdmission(
        "polynomial.rational.compute.partial_fraction_decomposition",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.partial_fractions",
    ),
    OperationAdmission(
        "polynomial.rational.compute.quotient_remainder",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic helper retained through the supported native API",
        native_symbol="jacobian.math.polynomials.divide",
    ),
    OperationAdmission(
        "polynomial.rational.multiply.compute",
        AdmissionDecision.KEEP,
        "distinct exact product of two bounded rational polynomials that sparse convolution cannot produce without caller reconstruction of like-term collection, zero cancellation, and canonical ordering",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
