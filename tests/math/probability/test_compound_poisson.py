"""Exact compound-Poisson cumulant tests."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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


def test_intensity_growth_is_admitted_before_cumulant_construction() -> None:
    large = 10**255 - 1
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(large)), probability=_q(Fraction(1))
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError):
        compound_poisson_cumulant_prefix(
            CompoundPoissonCumulantRequest(
                intensity=_q(Fraction(large)),
                jump_distribution=jumps,
                max_order=128,
            )
        )


def test_zero_intensity_and_signed_deterministic_jumps() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(-3, 2)), probability=_q(Fraction(1))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(
        CompoundPoissonCumulantRequest(
            intensity=_q(Fraction(2)), jump_distribution=jumps, max_order=4
        )
    )
    assert [row.jump_raw_moment.as_fraction() for row in result.cumulants] == [
        Fraction(-3, 2),
        Fraction(9, 4),
        Fraction(-27, 8),
        Fraction(81, 16),
    ]
    assert [row.cumulant.as_fraction() for row in result.cumulants] == [
        Fraction(-3),
        Fraction(9, 2),
        Fraction(-27, 4),
        Fraction(81, 8),
    ]

    zero = compound_poisson_cumulant_prefix(
        CompoundPoissonCumulantRequest(
            intensity=_q(Fraction()), jump_distribution=jumps, max_order=4
        )
    )
    assert all(row.cumulant.as_fraction() == 0 for row in zero.cumulants)


def test_native_boundary_rejects_non_request_values_with_owner_error() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compound_poisson_cumulant_prefix({})  # type: ignore[arg-type]
    assert (
        exc_info.value.errors()[0]["type"]
        == "probability.compound_poisson.request_type"
    )


def test_semantic_admission_rejects_negative_rate_and_unnormalized_jumps() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 2))
            ),
        )
    )
    with pytest.raises(OperationDomainValidationError) as rate_error:
        compound_poisson_cumulant_prefix(
            CompoundPoissonCumulantRequest(
                intensity=_q(Fraction(-1)), jump_distribution=jumps, max_order=1
            )
        )
    assert rate_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.nonnegative_intensity"
    )

    with pytest.raises(OperationDomainValidationError) as mass_error:
        compound_poisson_cumulant_prefix(
            CompoundPoissonCumulantRequest(
                intensity=_q(Fraction(1)), jump_distribution=jumps, max_order=1
            )
        )
    assert mass_error.value.errors()[0]["type"] == (
        "probability.compound_poisson.distribution_admission"
    )


def test_prefix_round_trips_with_its_source_parent() -> None:
    jumps = FiniteRationalDistribution(
        atoms=(
            FiniteDistributionAtom(
                value=_q(Fraction(0)), probability=_q(Fraction(1, 3))
            ),
            FiniteDistributionAtom(
                value=_q(Fraction(2)), probability=_q(Fraction(2, 3))
            ),
        )
    )
    result = compound_poisson_cumulant_prefix(
        CompoundPoissonCumulantRequest(
            intensity=_q(Fraction(5, 7)), jump_distribution=jumps, max_order=3
        )
    )
    assert result.model_validate_json(result.model_dump_json()) == result
    assert result.source.jump_distribution == jumps
