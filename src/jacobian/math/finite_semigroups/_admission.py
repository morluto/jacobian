"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_semigroups._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "semigroup.element.power_profile.compute",
        AdmissionDecision.KEEP,
        "exact power profile with index, period, idempotent, and cyclic subsemigroup",
    ),
    OperationAdmission(
        "semigroup.generated_subsemigroup.compute",
        AdmissionDecision.KEEP,
        "complete closure of generators under semigroup multiplication",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
