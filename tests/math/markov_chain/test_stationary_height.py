import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer
from jacobian.math.markov_chain._models import (
    StationaryDistributionRequest,
    TransitionMatrixRequest,
)


def _two_state_counterexample(exponent: int) -> dict[str, object]:
    two = 2**exponent
    three = 3**exponent
    p = (three - 1) // 2
    return {
        "matrix": [
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
        ]
    }


def test_stationary_request_rejects_exact_two_state_height_counterexample() -> None:
    with pytest.raises(ValidationError) as error:
        StationaryDistributionRequest.model_validate(_two_state_counterexample(45_000))
    assert (
        error.value.errors()[0]["type"]
        == "markov_chain.stationary_height_exceeds_bound"
    )


def test_stationary_bound_does_not_narrow_ergodic_decision_request() -> None:
    request = TransitionMatrixRequest.model_validate(_two_state_counterexample(45_000))

    assert len(request.matrix) == 2


def test_useful_near_boundary_stationary_request_remains_admitted() -> None:
    request = StationaryDistributionRequest.model_validate(
        _two_state_counterexample(4_000)
    )

    assert len(request.matrix) == 2
