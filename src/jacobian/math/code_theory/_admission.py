"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.code_theory._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "code.covering_radius.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.minimum_distance.compute",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
    OperationAdmission(
        "code.weight_distribution.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "code.dual_code.compute",
        AdmissionDecision.KEEP,
        "exact parity check matrix via null space computation over GF(p)",
    ),
    OperationAdmission(
        "code.syndrome.compute",
        AdmissionDecision.KEEP,
        "exact syndrome vector H*r^T mod p for received word decoding",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
