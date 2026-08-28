"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.optimization.submodular._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "combinatorics.set_function.evaluate",
        AdmissionDecision.DROP,
        "table lookup that merely echoes one caller-owned value",
    ),
    OperationAdmission(
        "combinatorics.set_function.monotonicity",
        AdmissionDecision.KEEP,
        "version-2 re-admission: kernel is the covering-relation scan (n*2^n exact "
        "inequalities) with the ground set widened to 16 and a 128-digit per-value "
        "height bound keeping every comparison on small big-ints; distinct exact "
        "bounded decision with materially wider leverage",
    ),
    OperationAdmission(
        "combinatorics.set_function.submodularity",
        AdmissionDecision.KEEP,
        "version-2 re-admission: kernel swapped from the O(4^n) all-pairs scan to "
        "the exact local characterization (C(n,2)*2^n inequalities) with the ground "
        "set widened to 16 and a 128-digit per-value height bound; distinct exact "
        "bounded decision with materially wider leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
