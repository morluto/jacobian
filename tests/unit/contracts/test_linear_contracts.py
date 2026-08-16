from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.linear import (
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
    LinearRationalSystem,
)
from jacobian.contracts.validated_analysis import RationalLinearProgramResult


def _system() -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "coefficients": {"entries": [[_q(2), _q(1)], [_q(1), _q(-1)]]},
        "rhs": [_q(5), _q(1)],
    }


def test_linear_system_requires_exact_matching_dimensions() -> None:
    system = LinearRationalSystem.model_validate(_system())
    assert system.variables == ("x", "y")
    assert len(system.coefficients.entries) == len(system.rhs) == 2

    malformed = _system()
    malformed["rhs"] = [_q(5)]
    with pytest.raises(ValidationError, match="right-hand side"):
        LinearRationalSystem.model_validate(malformed)

    malformed = _system()
    malformed["variables"] = ["x"]
    with pytest.raises(ValidationError, match="variable"):
        LinearRationalSystem.model_validate(malformed)


def test_linear_find_request_rejects_ambiguous_or_oversized_rationals() -> None:
    noncanonical = _system()
    noncanonical["rhs"] = [{"num": "2", "den": "2"}, _q(1)]
    with pytest.raises(ValidationError, match="reduced"):
        LinearRationalSolutionFindRequest.model_validate({"system": noncanonical})

    oversized = _system()
    oversized["rhs"] = [{"num": "1" * 257, "den": "1"}, _q(1)]
    with pytest.raises(ValidationError, match="256-digit bound"):
        LinearRationalSolutionFindRequest.model_validate({"system": oversized})


def test_inline_results_keep_only_mathematical_values() -> None:
    solution = LinearRationalSolutionResult(values=(_q(2), _q(1)))
    inconsistency = LinearRationalInconsistencyResult(
        left_witness=(_q(-2), _q(1)),
        rhs_pairing=_q(1),
    )

    assert solution.status == "SOLUTION"
    assert solution.values is not None
    assert inconsistency.status == "INCONSISTENT"
    assert inconsistency.left_witness is not None
    assert inconsistency.rhs_pairing is not None


def test_inline_results_preserve_completed_no_candidate_outcomes() -> None:
    solution = LinearRationalSolutionResult(status="INCONSISTENT")
    inconsistency = LinearRationalInconsistencyResult(status="CONSISTENT")

    assert solution.values is None
    assert inconsistency.left_witness is None
    with pytest.raises(ValidationError, match="agree with the result status"):
        LinearRationalSolutionResult(
            status="INCONSISTENT",
            values=(_q(2), _q(1)),
        )


def test_linear_program_outcomes_only_carry_their_mathematical_data() -> None:
    with pytest.raises(ValidationError, match="cannot carry a point"):
        RationalLinearProgramResult(
            status="INFEASIBLE",
            primal_candidate=(_q(1),),
        )
    with pytest.raises(ValidationError, match="only an optimal"):
        RationalLinearProgramResult(
            status="PRIMAL_FEASIBLE",
            primal_candidate=(_q(1),),
            primal_objective=_q(1),
            primal_residuals=(_q(0),),
            dual_candidate=(_q(1),),
        )
