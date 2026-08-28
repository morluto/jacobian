"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.diophantine_approximation._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "diophantine.continued_fraction.compute",
        AdmissionDecision.KEEP,
        "contract version 2 reevaluation: exact bounded periodic continued "
        "fraction of sqrt(D) for squarefree D; parsing retains only the "
        "bounded carrier and an explicit owner-local verifier replays a "
        "separately supplied claim within the admitted term envelope",
    ),
    OperationAdmission(
        "diophantine.convergents.compute",
        AdmissionDecision.NATIVE_ONLY,
        "contract version 2 reevaluation: deterministic projection of the "
        "retained continued-fraction result with a bounded canonical carrier; "
        "the explicit verifier replays the continuant recurrence of the same "
        "sqrt(D) under the derived digit cap",
        native_symbol="jacobian.math.number_theory.diophantine_approximation.convergents",
    ),
    OperationAdmission(
        "diophantine.pell_equation.solve",
        AdmissionDecision.KEEP,
        "distinct exact or explicitly bounded search outcome with material computational leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
