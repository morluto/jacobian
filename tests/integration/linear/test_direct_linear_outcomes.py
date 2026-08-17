from __future__ import annotations

import pytest
from tests.support.rationals import rational_payload as q

from jacobian.math.optimization._models import RationalLinearProgramRequest
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS
from jacobian.math.rational_linear._models import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.math.rational_linear._operations import (
    compute_rational_inconsistency,
    compute_rational_solution,
)

pytestmark = pytest.mark.requires_provider("flint")


def _system(rhs: list[dict[str, str]]) -> dict[str, object]:
    return {
        "variables": ["x"],
        "coefficients": {"entries": [[q(1)] for _ in rhs]},
        "rhs": rhs,
    }


def test_rational_linear_operations_return_mathematical_outcomes() -> None:
    consistent = _system([q(1)])
    inconsistent = _system([q(1), q(2)])

    solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": consistent})
    )
    no_solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": inconsistent})
    )
    consistency = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": consistent})
    )
    contradiction = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": inconsistent})
    )

    assert solution.status == "SOLUTION"
    assert solution.values is not None
    assert [v.model_dump(mode="json") for v in solution.values] == [q(1)]
    assert no_solution.status == "INCONSISTENT"
    assert consistency.status == "CONSISTENT"
    assert contradiction.status == "INCONSISTENT"
    assert contradiction.left_witness is not None
    assert contradiction.rhs_pairing is not None
    assert contradiction.rhs_pairing.model_dump(mode="json") == q(1)


def test_rational_linear_program_returns_an_optimum_not_a_certificate() -> None:
    operation = OPTIMIZATION_TOOLS[0]
    result = operation.run(
        RationalLinearProgramRequest.model_validate(
            {
                "program": {
                    "variables": ["x"],
                    "objective": [q(1)],
                    "coefficients": [[q(1)]],
                    "rhs": [q(1)],
                }
            }
        )
    )

    assert result.status == "OPTIMAL"
    # Guard the public wire shape: an optimum carries exactly the primal/dual
    # fields and no certificate, assurance, or other non-mathematical metadata.
    assert set(result.model_dump(mode="json")) == {
        "status",
        "primal_candidate",
        "dual_candidate",
        "primal_objective",
        "dual_objective",
        "primal_residuals",
        "dual_slacks",
    }
    assert [v.model_dump(mode="json") for v in result.primal_candidate] == [q(1)]
    assert [v.model_dump(mode="json") for v in result.dual_candidate] == [q(1)]
    assert result.primal_objective.model_dump(mode="json") == q(1)
    assert result.dual_objective.model_dump(mode="json") == q(1)
    assert [v.model_dump(mode="json") for v in result.primal_residuals] == [q(0)]
    assert [v.model_dump(mode="json") for v in result.dual_slacks] == [q(0)]


def test_rational_linear_program_handles_multiple_equalities() -> None:
    operation = OPTIMIZATION_TOOLS[0]
    result = operation.run(
        RationalLinearProgramRequest.model_validate(
            {
                "program": {
                    "variables": ["x", "y"],
                    "objective": [q(1), q(1)],
                    "coefficients": [[q(1), q(0)], [q(0), q(1)]],
                    "rhs": [q(1), q(2)],
                }
            }
        )
    )

    assert result.status == "OPTIMAL"
    assert [v.model_dump(mode="json") for v in result.primal_candidate] == [q(1), q(2)]
    assert [v.model_dump(mode="json") for v in result.primal_residuals] == [q(0), q(0)]
