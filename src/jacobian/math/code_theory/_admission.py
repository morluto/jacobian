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
        "exact covering radius bound to its retained source code and "
        "replayed by bounded syndrome-graph BFS within the declared "
        "state and transition bounds",
    ),
    OperationAdmission(
        "code.minimum_distance.compute",
        AdmissionDecision.KEEP,
        "exact minimum nonzero codeword weight bound to its retained "
        "source code and replayed by exact enumeration, including the "
        "documented zero-code length-n empty-code convention",
    ),
    OperationAdmission(
        "code.weight_distribution.compute",
        AdmissionDecision.KEEP,
        "exact codeword weight profile bound to its retained source "
        "code and replayed by exact enumeration with strictly ascending "
        "positive counts summing to q^rank",
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
