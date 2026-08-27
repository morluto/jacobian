"""Exact Poisson-binomial count distribution kernel using rational arithmetic."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.finite_stochastic_processes._poisson_binomial_models import (
    PoissonBinomialRequest,
    PoissonBinomialResult,
    _admit_probabilities,
)
from jacobian.math.finite_stochastic_processes.operations import (
    _poisson_binomial_kernel,
)


def compute_poisson_binomial(
    request: PoissonBinomialRequest,
) -> PoissonBinomialResult:
    """Compute the exact Poisson-binomial count distribution.

    Given independent Bernoulli trials with success probabilities
    p_1, ..., p_n (as canonical rationals), the Poisson-binomial
    distribution gives the probability of exactly k successes for k = 0, ..., n.

    Uses the direct recurrence with exact rational arithmetic:
    P(k, n) = P(k, n-1) * (1-p_n) + P(k-1, n-1) * p_n
    """
    return PoissonBinomialResult._from_kernel(
        request,
        count_distribution=_poisson_binomial_kernel(request.admission_plan),
    )


def verify_poisson_binomial_result(result: PoissonBinomialResult) -> bool:
    """Replay the bounded recurrence for an independently supplied result."""

    try:
        admission = _admit_probabilities(
            tuple(probability.as_fraction() for probability in result.probabilities)
        )
        expected = [Fraction(0)] * (len(admission.probabilities) + 1)
        expected[0] = Fraction(1)
        for probability in admission.probabilities:
            for index in range(len(expected) - 1, 0, -1):
                expected[index] = (
                    expected[index] * (1 - probability)
                    + expected[index - 1] * probability
                )
            expected[0] *= 1 - probability

        actual = tuple(
            atom.probability.as_fraction() for atom in result.count_distribution.atoms
        )
        return tuple(expected) == actual
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


__all__ = ["compute_poisson_binomial", "verify_poisson_binomial_result"]
