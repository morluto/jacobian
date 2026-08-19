"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import AdmissionDecision, OperationAdmission

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "semigroup.element.power.compute",
        AdmissionDecision.KEEP,
        "exact iterated power of one element with a positive exponent",
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
        "exact principal ideals {a} union {x*a, a*x : x in S} of requested elements",
    ),
)
