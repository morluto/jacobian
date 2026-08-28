"""Owner-local admission decisions for quadratic-form operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "quadratic_form.evaluate.compute",
        AdmissionDecision.KEEP,
        "exact source-bound rational-form evaluation over an explicit ordered axis; "
        "the direct Fraction kernel is complete under the entry-digit bounds and "
        "the active-term, aggregate-denominator, and exact-result budgets",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
