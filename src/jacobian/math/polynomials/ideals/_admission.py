"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.ideals._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.ideal.elimination.compute",
        AdmissionDecision.KEEP,
        "exact elimination ideal computation via lex Groebner basis extraction",
    ),
    OperationAdmission(
        "polynomial.ideal.groebner_basis.compute",
        AdmissionDecision.KEEP,
        "exact reduced Groebner basis computation via SymPy",
    ),
    OperationAdmission(
        "polynomial.ideal.minimal_primes.compute",
        AdmissionDecision.KEEP,
        "complete independently verified minimal-prime family over the declared rational coefficient field",
    ),
    OperationAdmission(
        "polynomial.ideal.normal_form.compute",
        AdmissionDecision.KEEP,
        "exact normal form reduction via Groebner basis remainder",
    ),
    OperationAdmission(
        "polynomial.ideal.quotient.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.ideal.radical.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.ideal.radical_membership.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "polynomial.ideal.saturation.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
