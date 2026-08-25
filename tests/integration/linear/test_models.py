from __future__ import annotations

import pytest
from tests.integration.linear._support import linear_validation_error
from tests.support.rationals import rational_payload as _q

from jacobian.math.matrices.rational_linear._models import (
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
    LinearRationalSystem,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
)


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
    with linear_validation_error():
        LinearRationalSystem.model_validate(malformed)

    malformed = _system()
    malformed["variables"] = ["x"]
    with linear_validation_error():
        LinearRationalSystem.model_validate(malformed)


def test_linear_find_request_rejects_ambiguous_or_oversized_rationals() -> None:
    noncanonical = _system()
    noncanonical["rhs"] = [{"num": "2", "den": "2"}, _q(1)]
    with linear_validation_error():
        LinearRationalSolutionFindRequest.model_validate({"system": noncanonical})

    oversized = _system()
    oversized["rhs"] = [{"num": "1" * 257, "den": "1"}, _q(1)]
    with linear_validation_error():
        LinearRationalSolutionFindRequest.model_validate({"system": oversized})


def test_inline_results_keep_only_mathematical_values() -> None:
    system = LinearRationalSystem.model_validate(_system())
    solution = LinearRationalSolutionResult(system=system, values=(_q(2), _q(1)))
    dependent = LinearRationalSystem.model_validate(
        {
            "variables": ["x", "y"],
            "coefficients": {"entries": [[_q(1), _q(1)], [_q(1), _q(1)]]},
            "rhs": [_q(0), _q(1)],
        }
    )
    inconsistency = LinearRationalInconsistencyResult(
        system=dependent,
        left_witness=(_q(-1), _q(1)),
        rhs_pairing=_q(1),
    )

    assert solution.status == "SOLUTION"
    assert solution.system == system
    assert solution.values is not None
    assert inconsistency.status == "INCONSISTENT"
    assert inconsistency.left_witness is not None
    assert inconsistency.rhs_pairing is not None


def test_inline_results_preserve_completed_no_candidate_outcomes() -> None:
    dependent = LinearRationalSystem.model_validate(
        {
            "variables": ["x", "y"],
            "coefficients": {"entries": [[_q(1), _q(1)], [_q(1), _q(1)]]},
            "rhs": [_q(0), _q(1)],
        }
    )
    solution = LinearRationalSolutionResult(system=dependent, status="INCONSISTENT")
    free = LinearRationalSystem.model_validate(
        {
            "variables": ["x", "y"],
            "coefficients": {"entries": [[_q(1), _q(1)]]},
            "rhs": [_q(1)],
        }
    )
    inconsistency = LinearRationalInconsistencyResult(
        system=free,
        status="CONSISTENT",
    )

    assert solution.values is None
    assert inconsistency.left_witness is None
    with linear_validation_error():
        LinearRationalSolutionResult(
            system=dependent,
            status="INCONSISTENT",
            values=(_q(2), _q(1)),
        )
    with linear_validation_error():
        LinearRationalInconsistencyResult(
            system=dependent,
            status="CONSISTENT",
            left_witness=(_q(-1), _q(1)),
            rhs_pairing=_q(1),
        )


def test_linear_program_outcomes_require_status_specific_source_bound_data() -> None:
    program = StandardFormRationalLinearProgram.model_validate(
        {
            "variables": ["x"],
            "objective": [_q(1)],
            "coefficients": [[_q(1)]],
            "rhs": [_q(1)],
        }
    )
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "program": program,
                "status": "INFEASIBLE",
                "primal_candidate": [_q(1)],
            }
        )
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "program": program,
                "status": "PRIMAL_FEASIBLE",
                "primal_candidate": [_q(1)],
                "primal_objective": _q(1),
                "primal_residuals": [_q(0)],
                "dual_candidate": [_q(1)],
            }
        )


def test_linear_program_raw_rational_bounds_precede_canonical_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian._exact as exact

    def fail_if_called(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("raw LP admission must run before integer parsing")

    monkeypatch.setattr(exact, "parse_canonical_integer", fail_if_called)
    oversized = "9" * 129
    with linear_validation_error():
        StandardFormRationalLinearProgram.model_validate(
            {
                "variables": ["x"],
                "objective": [{"num": oversized, "den": "1"}],
                "coefficients": [[_q(1)]],
                "rhs": [_q(1)],
            }
        )

    program = StandardFormRationalLinearProgram.model_construct(
        variables=("x",),
        objective=(_q(1),),
        coefficients=((_q(1),),),
        rhs=(_q(1),),
    )
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "program": program,
                "status": "UNKNOWN",
                "farkas_candidate": [{"num": "9" * 32_769, "den": "1"}],
            }
        )
