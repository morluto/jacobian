"""Owner-local admission for exact differential-operator application."""

from __future__ import annotations

from jacobian.catalog.admission import (
    AdmissionDecision,
    OperationAdmission,
    OperationRegistration,
)
from jacobian.math.polynomials.differential_operators._tools import TOOLS

ADMISSIONS: tuple[OperationAdmission, ...] = (
    OperationAdmission(
        "polynomial.differential_operator.apply.compute",
        AdmissionDecision.KEEP,
        (
            "one exact reusable action of a finite constant-coefficient "
            "multi-index operator on a canonical QQ polynomial; unlike the "
            "univariate derivative and fixed gradient/Laplacian projections, it "
            "preserves an arbitrary supplied operator and finite iterate while "
            "bounding its complete expansion, replay, coefficient growth, and output"
        ),
    ),
)

REGISTRATION = OperationRegistration(TOOLS, ADMISSIONS)

__all__ = ["REGISTRATION"]
