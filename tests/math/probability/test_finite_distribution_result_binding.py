"""Defining-invariant tests for trusted finite-distribution producers."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.probability._distribution import (
    FiniteDistributionAtom,
    FinitePushforwardMapEntry,
    FiniteRationalDistribution,
)
from jacobian.math.probability.operations import (
    condition,
    convolution,
    pushforward,
    raw_moment,
)


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational(num=str(numerator), den=str(denominator))


def _distribution() -> FiniteRationalDistribution:
    return FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(value=_q(0), probability=_q(1, 3)),
            FiniteDistributionAtom(value=_q(2), probability=_q(2, 3)),
        )
    )


def test_raw_moment_producer_satisfies_its_contribution_identity() -> None:
    source = _distribution()
    result = raw_moment(source.atoms, 2)
    assert all(
        item.powered_value.as_fraction() == item.value.as_fraction() ** result.order
        and item.contribution.as_fraction()
        == item.probability.as_fraction() * item.powered_value.as_fraction()
        for item in result.contributions
    )
    assert result.moment.as_fraction() == sum(
        (item.contribution.as_fraction() for item in result.contributions),
        start=0,
    )


def test_condition_producer_binds_distribution_to_contributions() -> None:
    source = _distribution()
    result = condition(source, (_q(0), _q(2)))
    event_probability = result.event_probability.as_fraction()
    assert all(
        item.conditioned_probability.as_fraction()
        == item.source_probability.as_fraction() / event_probability
        for item in result.contributions
    )
    assert tuple(
        (atom.value, atom.probability) for atom in result.distribution.atoms
    ) == tuple(
        (item.value, item.conditioned_probability) for item in result.contributions
    )


def test_pushforward_producer_aggregates_its_contributions() -> None:
    source = _distribution()
    result = pushforward(
        source,
        (
            FinitePushforwardMapEntry(source=_q(0), target=_q(1)),
            FinitePushforwardMapEntry(source=_q(2), target=_q(1)),
        ),
    )
    expected: dict[Fraction, Fraction] = {}
    for item in result.contributions:
        target = item.target.as_fraction()
        expected[target] = expected.get(target, 0) + item.probability.as_fraction()
    assert {
        atom.value.as_fraction(): atom.probability.as_fraction()
        for atom in result.distribution.atoms
    } == expected


def test_convolution_producer_builds_the_complete_product_measure() -> None:
    source = _distribution()
    result = convolution(source, source)
    assert all(
        item.sum_value.as_fraction()
        == item.left_value.as_fraction() + item.right_value.as_fraction()
        for item in result.contributions
    )
    assert len(result.contributions) == len(source.atoms) ** 2
