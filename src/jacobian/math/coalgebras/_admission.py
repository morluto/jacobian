"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.coalgebras._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "coalgebra.comultiplication.compute",
        AdmissionDecision.KEEP,
        "exact comultiplication Delta(c_i) computation over a prime field",
    ),
    OperationAdmission(
        "coalgebra.group_like_elements.compute",
        AdmissionDecision.KEEP,
        "exact group-like element search in a coalgebra over a prime field",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
