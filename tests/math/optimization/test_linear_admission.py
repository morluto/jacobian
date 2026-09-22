"""Derived LP limits, useful boundaries, and one request deadline."""

import json
from time import monotonic

import pytest
from pydantic import ValidationError
from tests.support.rationals import rational_payload as q

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization import general_linear_program, linear_program
from jacobian.math.optimization._general_models import GeneralFormRationalLinearProgram
from jacobian.math.optimization._linear_basis import (
    LINEAR_PROGRAM_WALL_SECONDS,
    admit_linear_program,
    basis_bounds,
)
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_BASES,
    MAX_RATIONAL_DIGITS,
    StandardFormRationalLinearProgram,
)


@pytest.mark.parametrize("sign", [-1, 1])
def test_scalar_envelope_agrees_for_native_and_json_components(sign: int) -> None:
    boundary = 10**MAX_RATIONAL_DIGITS
    native = {
        "variables": ["x"],
        "objective": [{"num": sign * (boundary - 1), "den": 1}],
        "coefficients": [[{"num": 1, "den": 1}]],
        "rhs": [{"num": 0, "den": 1}],
    }
    accepted = StandardFormRationalLinearProgram.model_validate(native)
    assert (
        StandardFormRationalLinearProgram.model_validate_json(
            accepted.model_dump_json()
        )
        == accepted
    )
    native["objective"] = [{"num": sign * boundary, "den": 1}]
    with pytest.raises(ValidationError, match="digit bound"):
        StandardFormRationalLinearProgram.model_validate(native)
    wire = accepted.model_dump(mode="json")
    wire["objective"] = [{"num": str(sign * boundary), "den": "1"}]
    with pytest.raises(ValidationError, match="digit bound"):
        StandardFormRationalLinearProgram.model_validate_json(json.dumps(wire))


def _dense_program(n: int, m: int) -> StandardFormRationalLinearProgram:
    return StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [f"x{i}" for i in range(n)],
                "objective": [q(1)] * n,
                "coefficients": [
                    [q(1 + int(i == j % m)) for j in range(n)] for i in range(m)
                ],
                "rhs": [q(1)] * m,
            }
        )
    )


def test_total_work_admission_rejects_18_by_6_before_search() -> None:
    with pytest.raises(OperationResourceAdmissionError) as caught:
        linear_program(_dense_program(18, 6))
    assert caught.value.errors()[0]["type"] == "optimization.linear.work_bound"


def test_total_work_boundary_accepts_14_by_8_shape() -> None:
    count, work = basis_bounds(14, 8)
    assert count < MAX_LINEAR_PROGRAM_BASES
    assert work + 16 * (8 + 1) * (14 + 1) < 50_000_000
    admission = admit_linear_program(_dense_program(14, 8))
    assert admission.initial_work < 50_000_000

    result = linear_program(_dense_program(14, 8))
    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.dual_objective is not None
    assert result.primal_objective.as_fraction().as_integer_ratio() == (8, 9)
    assert result.dual_objective.as_fraction().as_integer_ratio() == (8, 9)
    assert result.primal_candidate is not None
    assert result.dual_candidate is not None
    assert result.primal_residuals is not None
    assert result.dual_slacks is not None
    assert all(
        residual.as_fraction().as_integer_ratio() == (0, 1)
        for residual in result.primal_residuals
    )
    assert all(
        slack.as_fraction().as_integer_ratio() == (0, 1) for slack in result.dual_slacks
    )


def test_total_work_rejects_15_by_8_before_search() -> None:
    count, work = basis_bounds(15, 8)
    total_work = work + 16 * (8 + 1) * (15 + 1)
    assert count < MAX_LINEAR_PROGRAM_BASES
    assert total_work > 50_000_000
    program = _dense_program(15, 8)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        linear_program(program)
    assert caught.value.errors()[0]["type"] == "optimization.linear.work_bound"


def test_standard_basis_admission_reports_measured_costs() -> None:
    n, m = 24, 12
    program = _dense_program(n, m)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        linear_program(program)
    assert caught.value.errors()[0]["type"] == "optimization.linear.basis_bound"
    count, work = basis_bounds(n, m)
    work += 16 * (m + 1) * (n + 1)
    assert f"basis_estimate={count}" in str(caught.value)
    assert f"work_estimate={work}" in str(caught.value)
    assert "input_value" not in str(caught.value)


def test_work_bound_admission_precedes_late_search_exhaustion() -> None:
    n, m = 18, 6
    program = StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [f"x{i}" for i in range(n)],
                "objective": [q(1)] * n,
                "coefficients": [[q((j + 1) ** i) for j in range(n)] for i in range(m)],
                "rhs": [q(-1)] * m,
            }
        )
    )
    with pytest.raises(OperationResourceAdmissionError) as caught:
        linear_program(program)
    assert caught.value.errors()[0]["type"] == "optimization.linear.work_bound"


def test_native_general_deadline_covers_normalization_and_respects_outer_deadline() -> (
    None
):
    program = GeneralFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [
                    {"name": "x", "lower_bound": q(0)},
                    {"name": "y", "lower_bound": q(0)},
                ],
                "objective": {"sense": "MINIMIZE", "coefficients": [q(1), q(1)]},
                "constraints": [
                    {
                        "label": "sum",
                        "relation": "GE",
                        "coefficients": [q(1), q(1)],
                        "rhs": q(1),
                    }
                ],
            }
        )
    )
    start = monotonic()
    with request_execution(start):
        assert general_linear_program(program).status == "OPTIMAL"
        execution = current_request_execution()
        assert (
            execution is not None
            and execution.deadline == start + LINEAR_PROGRAM_WALL_SECONDS
        )
    with request_execution(start):
        bind_request_deadline(start - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            general_linear_program(program)


def test_rank_zero_maximum_shape_executes_without_empty_matrix_backend() -> None:
    program = StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": [f"x{i}" for i in range(32)],
                "objective": [q(0)] * 32,
                "coefficients": [[q(0)] * 32 for _ in range(64)],
                "rhs": [q(0)] * 64,
            }
        )
    )
    assert linear_program(program).status == "OPTIMAL"


def test_infeasible_component_overrides_unbounded_component() -> None:
    # The first block x0-x1=0 is unbounded for -x0; the second block x2=-1
    # is infeasible. A local ray alone cannot establish global unboundedness.
    program = StandardFormRationalLinearProgram.model_validate_json(
        json.dumps(
            {
                "variables": ["x0", "x1", "x2"],
                "objective": [q(-1), q(0), q(0)],
                "coefficients": [[q(1), q(-1), q(0)], [q(0), q(0), q(1)]],
                "rhs": [q(0), q(-1)],
            }
        )
    )
    result = linear_program(program)
    assert result.status == "INFEASIBLE"
    assert result.farkas_candidate is not None
    y = [v.as_fraction() for v in result.farkas_candidate]
    assert -y[1] < 0
    assert y[0] >= 0 and -y[0] >= 0 and y[1] >= 0
