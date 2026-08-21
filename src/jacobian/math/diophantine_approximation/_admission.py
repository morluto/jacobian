"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.diophantine_approximation._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "diophantine.continued_fraction.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "diophantine.convergents.compute",
        AdmissionDecision.NATIVE_ONLY,
        "useful deterministic projection of the retained continued-fraction result",
        native_symbol="jacobian.math.diophantine_approximation.convergents",
    ),
    OperationAdmission(
        "diophantine.pell_equation.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
