"""Derived LP limits, useful boundaries, and one request deadline."""

from time import monotonic

import pytest
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
    basis_bounds,
)
from jacobian.math.optimization._models import StandardFormRationalLinearProgram


@pytest.mark.parametrize(
    ("n", "m", "code"), [(18, 6, "work_bound"), (24, 12, "basis_bound")]
)
def test_standard_admission_reports_measured_costs(n: int, m: int, code: str) -> None:
    program = StandardFormRationalLinearProgram.model_validate(
        {
            "variables": [f"x{i}" for i in range(n)],
            "objective": [q(1)] * n,
            "coefficients": [
                [q(1 + int(i == j % m)) for j in range(n)] for i in range(m)
            ],
            "rhs": [q(1)] * m,
        }
    )
    with pytest.raises(OperationResourceAdmissionError) as caught:
        linear_program(program)
    assert caught.value.errors()[0]["type"] == f"optimization.linear.{code}"
    count, work = basis_bounds(n, m)
    work += 16 * (m + 1) * (n + 1)
    assert f"basis_estimate={count}" in str(caught.value)
    assert f"work_estimate={work}" in str(caught.value)
    assert "input_value" not in str(caught.value)


def test_native_general_deadline_covers_normalization_and_respects_outer_deadline() -> (
    None
):
    program = GeneralFormRationalLinearProgram.model_validate(
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
    program = StandardFormRationalLinearProgram.model_validate(
        {
            "variables": [f"x{i}" for i in range(32)],
            "objective": [q(0)] * 32,
            "coefficients": [[q(0)] * 32 for _ in range(64)],
            "rhs": [q(0)] * 64,
        }
    )
    assert linear_program(program).status == "OPTIMAL"


def test_disconnected_lp_returns_source_coordinate_optimum() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    n = 32
    program = {
        "variables": [f"x{i}" for i in range(n)],
        "objective": [q(1 + i % 2) for i in range(n)],
        "coefficients": [[q(int(i // 2 == j)) for i in range(n)] for j in range(16)],
        "rhs": [q(1)] * 16,
    }
    result = invoke_operation(
        "optimization.linear.rational_optimum.compute",
        {"program": program},
        Catalog.open(),
    ).output
    assert result["status"] == "OPTIMAL"
    assert result["primal_objective"] == q(16)
    assert result["primal_candidate"] == [q(1 - i % 2) for i in range(n)]
    assert result["dual_candidate"] == [q(1)] * 16
    assert result["dual_slacks"] == [q(i % 2) for i in range(n)]


def test_infeasible_component_overrides_unbounded_component() -> None:
    # The first block x0-x1=0 is unbounded for -x0; the second block x2=-1
    # is infeasible. A local ray alone cannot establish global unboundedness.
    program = StandardFormRationalLinearProgram.model_validate(
        {
            "variables": ["x0", "x1", "x2"],
            "objective": [q(-1), q(0), q(0)],
            "coefficients": [[q(1), q(-1), q(0)], [q(0), q(0), q(1)]],
            "rhs": [q(0), q(-1)],
        }
    )
    result = linear_program(program)
    assert result.status == "INFEASIBLE"
    assert result.farkas_candidate is not None
    y = [v.as_fraction() for v in result.farkas_candidate]
    assert -y[1] < 0
    assert y[0] >= 0 and -y[0] >= 0 and y[1] >= 0
