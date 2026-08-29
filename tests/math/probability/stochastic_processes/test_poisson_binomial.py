"""Tests for exact Poisson-binomial count distributions."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.stochastic_processes import poisson_binomial
from jacobian.math.probability.stochastic_processes._poisson_binomial_models import (
    PoissonBinomialRequest,
)
from jacobian.math.probability.stochastic_processes._tools import (
    compute_poisson_binomial,
)


def test_two_fair_coins() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(
                CanonicalRational.from_integer_ratio(1, 2),
                CanonicalRational.from_integer_ratio(1, 2),
            )
        )
    )
    dist = [atom.probability.as_fraction() for atom in result.count_distribution.atoms]
    assert dist == [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]


def test_native_api_returns_the_canonical_distribution() -> None:
    result = poisson_binomial(
        (
            CanonicalRational.from_integer_ratio(1, 2),
            CanonicalRational.from_integer_ratio(1, 3),
        )
    )
    assert [atom.probability.as_fraction() for atom in result.atoms] == [
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(1, 6),
    ]


def test_single_certain() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(1, 1),)
        )
    )
    dist = [atom.probability.as_fraction() for atom in result.count_distribution.atoms]
    assert dist == [Fraction(0), Fraction(1)]


def test_single_impossible() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(0, 1),)
        )
    )
    dist = [atom.probability.as_fraction() for atom in result.count_distribution.atoms]
    assert dist == [Fraction(1), Fraction(0)]


def test_three_fair_coins() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(
                CanonicalRational.from_integer_ratio(1, 2),
                CanonicalRational.from_integer_ratio(1, 2),
                CanonicalRational.from_integer_ratio(1, 2),
            )
        )
    )
    dist = [atom.probability.as_fraction() for atom in result.count_distribution.atoms]
    assert dist == [Fraction(1, 8), Fraction(3, 8), Fraction(3, 8), Fraction(1, 8)]
    # Sum should be 1
    assert sum(dist) == Fraction(1)


def test_native_admission_rejects_probabilities_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="closed unit interval"):
        poisson_binomial((CanonicalRational.from_integer_ratio(-1, 2),))
    with pytest.raises(ValueError, match="closed unit interval"):
        poisson_binomial((CanonicalRational.from_integer_ratio(2, 1),))


def test_operation_admission_uses_the_typed_domain_error() -> None:
    request = PoissonBinomialRequest(
        probabilities=(CanonicalRational.from_integer_ratio(-1, 2),)
    )

    with pytest.raises(OperationDomainValidationError) as caught:
        compute_poisson_binomial(request)

    assert caught.value.errors()[0]["loc"] == ("probabilities",)


def test_native_admission_rejects_result_digit_growth_before_execution() -> None:
    denominator = str(10**97 + 3)
    with pytest.raises(ValueError, match="exact result digit budget"):
        poisson_binomial((CanonicalRational(num="1", den=denominator),) * 45)


def test_native_result_probabilities_compose_into_request() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(
                CanonicalRational.from_integer_ratio(1, 2),
                CanonicalRational.from_integer_ratio(1, 3),
            )
        )
    )

    composed = compute_poisson_binomial(
        PoissonBinomialRequest(probabilities=result.probabilities)
    )

    assert composed == result


def test_serialized_result_probabilities_compose_into_request() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(1, 2),)
        )
    )
    payload = {"probabilities": result.model_dump(mode="json")["probabilities"]}

    request = PoissonBinomialRequest.model_validate(payload)

    assert request.probabilities == result.probabilities


def test_result_distribution_round_trips_into_finite_raw_moment() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(
                CanonicalRational.from_integer_ratio(1, 2),
                CanonicalRational.from_integer_ratio(1, 3),
            )
        )
    )
    restored = type(result).model_validate_json(result.model_dump_json())

    from jacobian.math.probability.operations import raw_moment

    moment = raw_moment(restored.count_distribution.atoms, 1)
    assert moment.moment.as_fraction() == Fraction(5, 6)
