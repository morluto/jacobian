"""Owner-local admission decisions for built-in math operations."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.finite_categories._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "finite_category.profile.compute",
        AdmissionDecision.KEEP,
        "exact hom-set, endomorphism, and identity profiles for a finite category",
    ),
    OperationAdmission(
        "finite_category.opposite.compute",
        AdmissionDecision.KEEP,
        "exact opposite category with reversed morphism directions",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
