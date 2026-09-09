"""Exact compound-Poisson cumulant tests."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.probability._compound_poisson import (
    CompoundPoissonCumulantRequest,
    compound_poisson_cumulant_prefix,
)
from jacobian.math.probability._distribution import (
    FiniteDistributionAtom,
    FiniteRationalDistribution,
)


def _q(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def test_compound_poisson_cumulants_are_intensity_times_jump_moments() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 2))
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(2)), probability=_q(Fraction(1, 2))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(
        CompoundPoissonCumulantRequest(
            intensity=_q(Fraction(3, 2)), jump_distribution=jumps, max_order=4
        )
    )
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [
        1,
        2,
        4,
        8,
    ]
    assert [row.cumulant.as_fraction() for row in result.cumulants] == [
        Fraction(3, 2),
        3,
        6,
        12,
    ]


def test_order_zero_returns_empty_prefix_without_a_pmf() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(value=_q(Fraction(1)), probability=_q(Fraction(1))),
        )
    )
    result = compound_poisson_cumulant_prefix(
        CompoundPoissonCumulantRequest(
            intensity=_q(Fraction()), jump_distribution=jumps, max_order=0
        )
    )
    assert result.cumulants == ()
