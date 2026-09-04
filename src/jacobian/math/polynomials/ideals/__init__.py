"""Exact polynomial-ideal values and operations."""

from jacobian.math.polynomials.ideals._models import (
    GradedBettiNumber,
    IdealComputationBudget,
    IdealContainmentLedger,
    IdealContainmentResult,
    IdealEqualityResult,
    IdealMembershipCertificateResult,
    LcmLatticeHomologyEntry,
    MonomialIdealBettiResult,
    MultigradedBettiNumber,
)
from jacobian.math.polynomials.ideals.operations import (
    monomial_ideal_graded_betti_table,
)


def __getattr__(name: str) -> object:
    if name not in {"ideal_containment", "ideal_equality"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from jacobian.math.polynomials.ideals import operations

    value = getattr(operations, name)
    globals()[name] = value
    return value


__all__ = [
    "GradedBettiNumber",
    "IdealComputationBudget",
    "IdealContainmentLedger",
    "IdealContainmentResult",
    "IdealEqualityResult",
    "IdealMembershipCertificateResult",
    "LcmLatticeHomologyEntry",
    "MonomialIdealBettiResult",
    "MultigradedBettiNumber",
    "monomial_ideal_graded_betti_table",
]
