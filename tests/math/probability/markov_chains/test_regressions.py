from __future__ import annotations

import sys
from fractions import Fraction
from types import FrameType

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.markov_chains import (
    ergodic_properties,
    stationary_distribution,
    stationary_distribution_result,
)
from jacobian.math.probability.markov_chains._models import (
    StationaryDistributionRequest,
    TransitionMatrixRequest,
)
from jacobian.math.probability.markov_chains._tools import (
    compute_ergodic_decision,
    compute_stationary_distribution,
)


def test_native_namespace_exposes_values_and_kernels_not_wire_requests() -> None:
    import jacobian.math.probability.markov_chains as markov_chain

    assert "StationaryDistributionRequest" not in markov_chain.__all__
    assert "TransitionMatrixRequest" not in markov_chain.__all__
    assert "TransitionMatrix" in markov_chain.__all__
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


def _lazy_cycle_request(size: int) -> dict[str, object]:
    return {
        "matrix": [
            [
                {
                    "num": "1" if column in {row, (row + 1) % size} else "0",
                    "den": "2" if column in {row, (row + 1) % size} else "1",
                }
                for column in range(size)
            ]
            for row in range(size)
        ]
    }


def test_flint_stationary_solve_exceeds_the_previous_state_ceiling() -> None:
    size = 64
    request = StationaryDistributionRequest.model_validate(_lazy_cycle_request(size))

    result = compute_stationary_distribution(request)

    assert result.unique is True
    distribution = tuple(
        value.as_fraction() for value in result.extreme_distributions[0].distribution
    )
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    assert distribution == (Fraction(1, size),) * size
    assert sum(distribution) == 1
    assert all(value >= 0 for value in distribution)
    assert (
        tuple(
            sum(distribution[row] * matrix[row][column] for row in range(size))
            for column in range(size)
        )
        == distribution
    )


def test_stationary_solve_work_rejects_before_flint() -> None:
    request = StationaryDistributionRequest.model_validate(_lazy_cycle_request(101))

    with pytest.raises(OperationDomainValidationError, match="solve-work bound"):
        compute_stationary_distribution(request)


def test_generic_markov_requests_keep_the_32_state_carrier() -> None:
    with pytest.raises(ValidationError):
        TransitionMatrixRequest.model_validate(_lazy_cycle_request(33))


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
    ("matrix", "error_code"),
    [
        (
            [[{"num": "1", "den": "1"}], [{"num": "0", "den": "1"}]],
            "markov_chain.transition_matrix_not_square",
        ),
        (
            [[{"num": "2", "den": "1"}]],
            "markov_chain.transition_row_not_stochastic",
        ),
        (
            [[{"num": "-1", "den": "1"}]],
            "markov_chain.transition_probability_negative",
        ),
    ],
)
def test_transition_contract_rejects_non_stochastic_matrices(
    matrix: object, error_code: str
) -> None:
    if error_code == "markov_chain.transition_matrix_not_square":
        with pytest.raises(ValidationError) as structural_error:
            TransitionMatrixRequest.model_validate({"matrix": matrix})
        assert structural_error.value.errors()[0]["type"] == error_code
    else:
        request = TransitionMatrixRequest.model_validate({"matrix": matrix})
        with pytest.raises(OperationDomainValidationError) as domain_error:
            compute_ergodic_decision(request)
        assert domain_error.value.errors()[0]["type"] == error_code


@pytest.mark.parametrize("class_size", [1, 2])
def test_stationary_family_admits_many_small_closed_classes(class_size: int) -> None:
    from jacobian.math.probability.markov_chains.values import MAX_STATIONARY_STATES

    size = MAX_STATIONARY_STATES
    matrix = tuple(
        tuple(
            Fraction(1, class_size)
            if i // class_size == j // class_size
            else Fraction()
            for j in range(size)
        )
        for i in range(size)
    )
    result = stationary_distribution_result(matrix)
    assert len(result.extreme_distributions) == size // class_size
    for index, extreme in enumerate(result.extreme_distributions):
        states = tuple(range(index * class_size, (index + 1) * class_size))
        assert extreme.closed_class == states
        expected = tuple(
            Fraction(1, class_size) if j in states else Fraction() for j in range(size)
        )
        assert tuple(value.as_fraction() for value in extreme.distribution) == expected


def test_stationary_admission_does_not_charge_transient_states_as_solve_rows() -> None:
    from jacobian.math.probability.markov_chains.values import MAX_STATIONARY_STATES

    size = MAX_STATIONARY_STATES
    matrix = tuple(
        tuple(Fraction(1, 2) if j >= size - 2 else Fraction() for j in range(size))
        for _ in range(size)
    )
    assert stationary_distribution(matrix) == matrix[0]


def test_stationary_class_decomposition_and_solve_work_are_not_repeated() -> None:
    from tests.fixtures.accounting import assert_charged_work_parity

    from jacobian.math.probability.markov_chains import operations
    from jacobian.math.probability.markov_chains._flint import solve_stationary_class
    from jacobian.math.probability.markov_chains.values import MAX_STATIONARY_STATES

    size = MAX_STATIONARY_STATES
    matrix = tuple(
        tuple(Fraction(int(i == j)) for j in range(size)) for i in range(size)
    )
    executed = {"support_cells": 0, "solve_cubes": 0}

    def observe(frame: FrameType, event: str, _arg: object) -> None:
        if event != "call":
            return
        if frame.f_code is operations._closed_communicating_classes.__code__:
            executed["support_cells"] += len(frame.f_locals["matrix"]) ** 2
        elif frame.f_code is solve_stationary_class.__code__:
            executed["solve_cubes"] += len(frame.f_locals["closed_class"]) ** 3

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        result = stationary_distribution_result(matrix)
        serialized = result.model_dump_json()
        type(result).model_validate_json(serialized)
    finally:
        sys.setprofile(previous)

    charged = {"support_cells": size**2, "solve_cubes": size}
    assert executed == charged
    assert_charged_work_parity(charged=charged, executed=executed)
    assert len(result.extreme_distributions) == size


@pytest.mark.parametrize(("first_size", "admitted"), [(99, True), (100, False)])
def test_stationary_sums_work_across_closed_classes(
    first_size: int, admitted: bool
) -> None:
    from jacobian.math.probability.markov_chains.values import MAX_STATIONARY_STATES

    size = MAX_STATIONARY_STATES
    matrix = tuple(
        tuple(
            Fraction(1, first_size if i < first_size else size - first_size)
            if (i < first_size) == (j < first_size)
            else Fraction()
            for j in range(size)
        )
        for i in range(size)
    )
    if not admitted:
        with pytest.raises(OperationDomainValidationError, match="solve-work bound"):
            stationary_distribution_result(matrix)
        return
    result = stationary_distribution_result(matrix)
    assert [item.closed_class for item in result.extreme_distributions] == [
        tuple(range(first_size)),
        tuple(range(first_size, size)),
    ]
    for state, extreme in zip(
        (0, first_size), result.extreme_distributions, strict=True
    ):
        assert (
            tuple(value.as_fraction() for value in extreme.distribution)
            == matrix[state]
        )
