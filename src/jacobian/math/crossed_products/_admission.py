"""Owner-local catalog admission for finite-coset crossed products."""

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.crossed_products._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "crossed_product.multiply.compute",
        AdmissionDecision.KEEP,
        "exact finite-support convolution in a fully validated F_p[Z^d x_c Q] presentation",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["ADMISSIONS", "REGISTRATION"]
