from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.markov_chain import (
    ergodic_properties,
    stationary_distribution,
)
from jacobian.math.markov_chain._models import (
    StationaryDistributionRequest,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain._operations import (
    _verify_stationary_distribution_result,
    compute_ergodic_decision,
    compute_stationary_distribution,
)


def test_native_namespace_exposes_values_and_kernels_not_wire_requests() -> None:
    import jacobian.math.markov_chain as markov_chain

    assert "StationaryDistributionRequest" not in markov_chain.__all__
    assert "TransitionMatrixRequest" not in markov_chain.__all__
    assert markov_chain.stationary_distribution is stationary_distribution


def test_ergodicity_uses_irreducibility_and_period_not_square_positivity() -> None:
    result = compute_ergodic_decision(
        TransitionMatrixRequest.model_validate(
            {
                "matrix": [
                    [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    [
                        {"num": "0", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    [
                        {"num": "1", "den": "2"},
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "2"},
                    ],
                ]
            }
        )
    )

    assert result.is_irreducible is True
    assert result.is_aperiodic is True
    assert result.is_ergodic is True


def test_aperiodicity_is_checked_for_each_communicating_class() -> None:
    result = compute_ergodic_decision(
        TransitionMatrixRequest.model_validate(
            {
                "matrix": [
                    [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                    [
                        {"num": "0", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                ]
            }
        )
    )

    assert result.is_irreducible is False
    assert result.is_aperiodic is True
    assert result.is_ergodic is False


def test_stationary_family_exposes_every_closed_class() -> None:
    request = StationaryDistributionRequest.model_validate(
        {
            "matrix": [
                [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "2"},
                    {"num": "1", "den": "2"},
                ],
                [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                ],
            ]
        }
    )

    result = compute_stationary_distribution(request)

    assert result.transition_matrix == request.matrix
    assert _verify_stationary_distribution_result(result)
    assert not _verify_stationary_distribution_result(
        result.model_copy(update={"unique": not result.unique})
    )

    assert result.unique is False
    assert [item.closed_class for item in result.extreme_distributions] == [(1,), (2,)]
    assert [
        [value.as_fraction() for value in item.distribution]
        for item in result.extreme_distributions
    ] == [[0, 1, 0], [0, 0, 1]]


def test_native_singular_stationary_helper_rejects_nonunique_chain() -> None:
    request = StationaryDistributionRequest.model_validate(
        {
            "matrix": [
                [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            ]
        }
    )

    with pytest.raises(ValueError, match="does not have a unique"):
        stationary_distribution(
            tuple(tuple(value.as_fraction() for value in row) for row in request.matrix)
        )


def test_native_markov_api_accepts_canonical_fraction_matrices() -> None:
    request = StationaryDistributionRequest.model_validate(
        {
            "matrix": [
                [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
            ]
        }
    )

    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    assert stationary_distribution(matrix) == (Fraction(1, 2), Fraction(1, 2))
    assert ergodic_properties(matrix) == (True, True)


def test_stationary_family_solves_each_nonsingleton_closed_class_exactly() -> None:
    request = StationaryDistributionRequest.model_validate(
        {
            "matrix": [
                [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "2"},
                    {"num": "1", "den": "2"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "4"},
                    {"num": "3", "den": "4"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "1", "den": "3"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "2", "den": "3"},
                    {"num": "0", "den": "1"},
                ],
            ]
        }
    )

    result = compute_stationary_distribution(request)

    assert [item.closed_class for item in result.extreme_distributions] == [
        (0, 1),
        (2, 3),
    ]
    assert [
        [value.as_fraction() for value in item.distribution]
        for item in result.extreme_distributions
    ] == [
        [Fraction(1, 2), Fraction(1, 2), 0, 0, 0],
        [0, 0, Fraction(1, 3), Fraction(2, 3), 0],
    ]


@pytest.mark.parametrize(
    "matrix",
    [
        [[{"num": "1", "den": "1"}], [{"num": "0", "den": "1"}]],
        [[{"num": "2", "den": "1"}]],
        [[{"num": "-1", "den": "1"}]],
    ],
)
def test_transition_contract_rejects_non_stochastic_matrices(matrix: object) -> None:
    with pytest.raises(ValidationError) as error:
        TransitionMatrixRequest.model_validate({"matrix": matrix})
    assert error.value.errors()[0]["type"] in {
        "markov_chain.transition_matrix_not_square",
        "markov_chain.transition_probability_negative",
        "markov_chain.transition_row_not_stochastic",
    }
