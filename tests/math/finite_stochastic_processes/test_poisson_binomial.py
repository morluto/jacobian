"""Tests for exact Poisson-binomial count distributions."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.finite_stochastic_processes import poisson_binomial
from jacobian.math.finite_stochastic_processes._poisson_binomial_models import (
    PoissonBinomialRequest,
)
from jacobian.math.finite_stochastic_processes._poisson_binomial_operations import (
    compute_poisson_binomial,
    verify_poisson_binomial_result,
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


def test_catalog_dispatch_reuses_request_admission_for_exact_result() -> None:
    result = invoke_operation(
        "probability.poisson_binomial.distribution.compute",
        {
            "probabilities": [
                {"num": "1", "den": "2"},
                {"num": "1", "den": "3"},
            ]
        },
        Catalog.open(),
    )

    assert result.output == {
        "probabilities": [
            {"num": "1", "den": "2"},
            {"num": "1", "den": "3"},
        ],
        "count_distribution": {
            "atoms": [
                {
                    "value": {"num": "0", "den": "1"},
                    "probability": {"num": "1", "den": "3"},
                },
                {
                    "value": {"num": "1", "den": "1"},
                    "probability": {"num": "1", "den": "2"},
                },
                {
                    "value": {"num": "2", "den": "1"},
                    "probability": {"num": "1", "den": "6"},
                },
            ]
        },
    }


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


def test_rejects_probabilities_outside_the_unit_interval() -> None:
    with pytest.raises(ValidationError, match="closed unit interval"):
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(-1, 2),)
        )
    with pytest.raises(ValidationError, match="closed unit interval"):
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(2, 1),)
        )


def test_rejects_result_digit_growth_before_execution() -> None:
    denominator = str(10**97 + 3)
    with pytest.raises(ValidationError, match="exact result digit budget"):
        PoissonBinomialRequest(
            probabilities=(CanonicalRational(num="1", den=denominator),) * 45
        )


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


def test_result_verifier_rejects_forged_serialized_distribution_masses() -> None:
    result = compute_poisson_binomial(
        PoissonBinomialRequest(
            probabilities=(CanonicalRational.from_integer_ratio(1, 2),)
        )
    )
    assert verify_poisson_binomial_result(result)

    payload = result.model_dump(mode="json")
    payload["count_distribution"]["atoms"] = [
        {
            "value": {"num": "0", "den": "1"},
            "probability": {"num": "1", "den": "1"},
        },
        {
            "value": {"num": "1", "den": "1"},
            "probability": {"num": "0", "den": "1"},
        },
    ]
    forged = type(result).model_validate(payload)

    assert not verify_poisson_binomial_result(forged)


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

    from jacobian.math.probability._distribution import FiniteRawMomentRequest
    from jacobian.math.probability._operations import _raw_moment

    moment = _raw_moment(
        FiniteRawMomentRequest(atoms=restored.count_distribution.atoms, order=1)
    )
    assert moment.moment.as_fraction() == Fraction(5, 6)
