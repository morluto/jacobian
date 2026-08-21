"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_semigroups._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "semigroup.element.power.compute",
        AdmissionDecision.KEEP,
        "exact positive power reduced through the finite power profile",
    ),
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
    OperationAdmission(
        "semigroup.idempotents.compute",
        AdmissionDecision.KEEP,
        "exact set of idempotent elements e with e*e = e",
    ),
    OperationAdmission(
        "semigroup.principal_ideals.compute",
        AdmissionDecision.KEEP,
        "exact principal two-sided ideals S^1 a S^1 of requested elements",
    ),
    OperationAdmission(
        "semigroup.green_relations.compute",
        AdmissionDecision.KEEP,
        "exact Green relations L, R, H, D, J via principal ideal equality",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
