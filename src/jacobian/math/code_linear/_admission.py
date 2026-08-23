"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.code_linear._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "code.linear.received_word_profile.compute",
        AdmissionDecision.KEEP,
        "complete source-bound coset weight distribution with optional replayable threshold witnesses under explicit enumeration and output bounds",
    ),
    OperationAdmission(
        "code.linear.codeword.check",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.dual.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.equal.decide",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.from_generator.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.macwilliams_transform.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.parity_check.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.puncture.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.shorten.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.linear.syndrome.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
