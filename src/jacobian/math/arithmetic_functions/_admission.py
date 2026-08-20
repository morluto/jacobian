"""Owner-local admission decisions for built-in math operations."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.arithmetic_functions._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "arithmetic.dirichlet_convolution.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "arithmetic.dirichlet_inverse.compute",
        AdmissionDecision.KEEP,
        "distinct exact bounded mathematical value or invariant with material computational or reliability leverage",
    ),
    OperationAdmission(
        "arithmetic.mobius_transform.compute",
        AdmissionDecision.KEEP,
        "reusable typed mathematical construction or transformation with a distinct discovery intent",
    ),
    OperationAdmission(
        "arithmetic.summatory_function.compute",
        AdmissionDecision.DROP,
        "ordinary finite prefix sum without catalog-level leverage",
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)
