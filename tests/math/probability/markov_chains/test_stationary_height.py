import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import rational_matrix_from_fractions
from jacobian.math.probability.markov_chains import stationary_distribution_result
from jacobian.math.probability.markov_chains._models import (
    StationaryDistributionRequest,
    TransitionMatrixRequest,
)
from jacobian.math.probability.markov_chains._tools import (
    compute_stationary_distribution,
)


def _two_state_counterexample(exponent: int) -> dict[str, object]:
    two = 2**exponent
    three = 3**exponent
    p = (three - 1) // 2
    return {
        "matrix": {
            "domain": "QQ",
            "entries": [
                [
                    {
                        "num": format_canonical_integer(two - 1),
                        "den": format_canonical_integer(two),
                    },
                    {"num": "1", "den": format_canonical_integer(two)},
                ],
                [
                    {
                        "num": format_canonical_integer(p),
                        "den": format_canonical_integer(three),
                    },
                    {
                        "num": format_canonical_integer(three - p),
                        "den": format_canonical_integer(three),
                    },
                ],
            ],
        }
    }


def test_stationary_request_rejects_exact_two_state_height_counterexample() -> None:
    request = StationaryDistributionRequest.model_validate_json(
        json.dumps(_two_state_counterexample(45_000))
    )

    with pytest.raises(OperationDomainValidationError) as error:
        compute_stationary_distribution(request)
    assert (
        error.value.errors()[0]["type"]
        == "markov_chain.stationary_height_exceeds_bound"
    )


def test_stationary_bound_does_not_narrow_ergodic_decision_request() -> None:
    request = TransitionMatrixRequest.model_validate_json(
        json.dumps(_two_state_counterexample(45_000))
    )

    assert len(request.matrix.entries) == 2


def test_useful_near_boundary_stationary_request_remains_admitted() -> None:
    request = StationaryDistributionRequest.model_validate_json(
        json.dumps(_two_state_counterexample(4_000))
    )

    assert len(request.matrix.entries) == 2


@pytest.mark.parametrize("exponent", [100, MAX_CANONICAL_RATIONAL_DIGITS - 1])
def test_transient_height_is_only_charged_for_retaining_the_source(
    exponent: int,
) -> None:
    p = Fraction(1, 10**exponent)
    matrix = (
        (Fraction(), p, 1 - p),
        (Fraction(), Fraction(1), Fraction()),
        (Fraction(), Fraction(), Fraction(1)),
    )
    result = stationary_distribution_result(rational_matrix_from_fractions(matrix))
    assert [item.closed_class for item in result.extreme_distributions] == [(1,), (2,)]
    assert (
        tuple(value.as_fraction() for value in result.transition_matrix.entries[0])
        == matrix[0]
    )
    type(result).model_validate_json(result.model_dump_json())


def test_native_stationary_rejects_unrepresentable_transient_source() -> None:
    p = Fraction(1, 10**MAX_CANONICAL_RATIONAL_DIGITS)
    matrix = (
        (Fraction(), p, 1 - p),
        (Fraction(), Fraction(1), Fraction()),
        (Fraction(), Fraction(), Fraction(1)),
    )
    # The canonical carrier rejects an unrepresentable source before the
    # operation can be admitted; this is the public boundary for native APIs.
    with pytest.raises(ValidationError) as error:
        stationary_distribution_result(rational_matrix_from_fractions(matrix))
    assert error.value.errors()[0]["type"] == "exact_integer.digit_bound"
