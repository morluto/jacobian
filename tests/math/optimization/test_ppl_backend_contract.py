"""Independent exact checks at the private PPL result boundary."""

import json
from fractions import Fraction

import pytest
from tests.support.rationals import rational_payload as q

from jacobian.math.optimization import linear_program
from jacobian.math.optimization import operations as solver
from jacobian.math.optimization._models import StandardFormRationalLinearProgram
from jacobian.math.optimization._ppl import ExactLinearOutcome


def _program() -> StandardFormRationalLinearProgram:
    return StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": ["x"],
                "objective": [q(1)],
                "coefficients": [[q(1)]],
                "rhs": [q(1)],
            }
        )
    )


@pytest.mark.parametrize(
    "outcome",
    [
        ExactLinearOutcome(
            status="OPTIMAL",
            point=(Fraction(2),),
            dual=(Fraction(1),),
        ),
        ExactLinearOutcome(
            status="INFEASIBLE",
            witness=(Fraction(1),),
        ),
        ExactLinearOutcome(
            status="UNBOUNDED",
            point=(Fraction(1),),
            ray=(Fraction(1),),
        ),
    ],
)
def test_invalid_backend_claim_never_becomes_a_mathematical_result(
    monkeypatch: pytest.MonkeyPatch,
    outcome: ExactLinearOutcome,
) -> None:
    monkeypatch.setattr(solver, "solve_standard_form_process", lambda *args: outcome)
    with pytest.raises(RuntimeError, match="produced no mathematical result"):
        linear_program(_program())
