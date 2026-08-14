from __future__ import annotations

import pytest
from tests.support.rationals import rational_payload as q

from jacobian.contracts.linear import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.contracts.validated_analysis import RationalLinearProgramRequest
from jacobian.domains.optimization import rational_optimization_operations
from jacobian.domains.rational_linear.operations import (
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
    assert solution.model_dump(mode="json")["values"] == [q(1)]
    assert no_solution.status == "INCONSISTENT"
    assert consistency.status == "CONSISTENT"
    assert contradiction.status == "INCONSISTENT"
    assert contradiction.left_witness is not None
    assert contradiction.model_dump(mode="json")["rhs_pairing"] == q(1)


def test_rational_linear_program_returns_an_optimum_not_a_certificate() -> None:
    operation = rational_optimization_operations()[0]
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
    assert result.model_dump(mode="json") == {
        "status": "OPTIMAL",
        "primal_candidate": [q(1)],
        "dual_candidate": [q(1)],
        "primal_objective": q(1),
        "dual_objective": q(1),
        "primal_residuals": [q(0)],
        "dual_slacks": [q(0)],
    }


def test_rational_linear_program_handles_multiple_equalities() -> None:
    operation = rational_optimization_operations()[0]
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
    assert result.model_dump(mode="json")["primal_candidate"] == [q(1), q(2)]
    assert result.model_dump(mode="json")["primal_residuals"] == [q(0), q(0)]
