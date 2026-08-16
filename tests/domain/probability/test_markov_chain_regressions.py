from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.markov_chain import TransitionMatrixRequest
from jacobian.domains.markov_chain.operations import compute_ergodic_decision


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


@pytest.mark.parametrize(
    "matrix",
    [
        [[{"num": "1", "den": "1"}], [{"num": "0", "den": "1"}]],
        [[{"num": "2", "den": "1"}]],
        [[{"num": "-1", "den": "1"}]],
    ],
)
def test_transition_contract_rejects_non_stochastic_matrices(matrix: object) -> None:
    with pytest.raises(ValidationError):
        TransitionMatrixRequest.model_validate({"matrix": matrix})
