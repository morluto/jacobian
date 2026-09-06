"""Operation-owned probability and partition admission."""

from fractions import Fraction

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.stochastic_processes.values import (
    FiniteProbabilitySpace,
    FiniteSigmaAlgebra,
)


def _reject(reason: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("space",),
        code=f"finite_stochastic_process.{reason}",
        message=message,
    )


def admit_probability_space(space: FiniteProbabilitySpace) -> None:
    """Check at most 64 masses with 256-digit components.

    A common denominator has at most 16384 digits; positive partial sums
    require at most 16386 numerator digits. No backend or unbounded expansion
    is needed, and callers share this check within one composite execution.
    """
    total = Fraction(0)
    for mass in space.masses:
        value = mass.as_fraction()
        if value <= 0:
            _reject("mass_nonpositive", "masses must be positive")
        total += value
    if total != 1:
        _reject("mass_sum_invalid", "masses must sum to exactly 1")


def admit_partition(sigma: FiniteSigmaAlgebra) -> None:
    """Check disjoint coverage in at most 64 squared declared memberships.

    Probability normalization is admitted separately once by the caller.
    """
    seen: set[str] = set()
    for block in sigma.blocks:
        for sample in block:
            if sample in seen:
                _reject("partition_blocks_overlap", "partition blocks must be disjoint")
            seen.add(sample)
    if seen != set(sigma.space.samples):
        _reject("partition_incomplete", "blocks must partition the entire sample space")
