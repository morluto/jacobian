"""Source-coordinate certificates outgrow the independent solver envelope."""

from fractions import Fraction
from itertools import combinations
from typing import Any

import pytest

from jacobian.math.optimization._optimality import (
    RationalLinearOptimalityCandidate,
    RationalLinearOptimalityResult,
    check_linear_optimality,
)


def certificate(rows: int, *, standard: bool = False) -> dict[str, Any]:
    n = 15
    cyclic = [tuple(sorted((i + j) % n for j in range(4))) for i in range(n)]
    supports = (
        cyclic + [s for s in combinations(range(n), 4) if s not in cyclic][: rows - n]
    )

    def r(num: int, den: int = 1) -> dict[str, int]:
        return {"num": num, "den": den}

    matrix = [[r(int(i in s)) for i in range(n)] for s in supports]
    if standard:
        program = {
            "variables": [f"x{i}" for i in range(n)],
            "objective": [r(1)] * n,
            "coefficients": matrix,
            "rhs": [r(1)] * rows,
        }
    else:
        program = {
            "variables": [{"name": f"x{i}", "lower_bound": r(0)} for i in range(n)],
            "objective": {"sense": "MINIMIZE", "coefficients": [r(1)] * n},
            "constraints": [
                {"label": f"row{i}", "coefficients": a, "relation": "GE", "rhs": r(1)}
                for i, a in enumerate(matrix)
            ],
        }
    return {
        "program": program,
        "primal_candidate": [r(1, 4)] * n,
        "constraint_dual": [r(1, 4)] * n + [r(0)] * (rows - n),
        "lower_bound_dual": [r(0)] * n,
        "upper_bound_dual": [r(0)] * n,
    }


@pytest.mark.parametrize("rows", [65, 246])
@pytest.mark.parametrize("standard", [False, True])
def test_large_supplied_certificate(rows: int, standard: bool) -> None:
    candidate = RationalLinearOptimalityCandidate.model_validate(
        certificate(rows, standard=standard)
    )
    result = check_linear_optimality(candidate)
    assert result.is_optimal
    assert result.primal_objective.as_fraction() == Fraction(15, 4)
    assert result.dual_objective == result.primal_objective
    assert len(result.primal_residuals) == rows
    assert (
        RationalLinearOptimalityResult.model_validate_json(result.model_dump_json())
        == result
    )


def zero_certificate(n: int, m: int, *, standard: bool = False) -> dict[str, Any]:
    r = {"num": 0, "den": 1}
    if standard:
        program = {
            "variables": [f"x{i}" for i in range(n)],
            "objective": [r] * n,
            "coefficients": [[r] * n] * m,
            "rhs": [r] * m,
        }
    else:
        program = {
            "variables": [{"name": f"x{i}", "lower_bound": r} for i in range(n)],
            "objective": {"sense": "MINIMIZE", "coefficients": [r] * n},
            "constraints": [
                {
                    "label": f"row{i}",
                    "coefficients": [r] * n,
                    "relation": "EQ",
                    "rhs": r,
                }
                for i in range(m)
            ],
        }
    return {
        "program": program,
        "primal_candidate": [r] * n,
        "constraint_dual": [r] * m,
        "lower_bound_dual": [r] * n,
        "upper_bound_dual": [r] * n,
    }


@pytest.mark.parametrize("standard", [False, True])
def test_every_late_row_and_dual_entry_is_checked(standard: bool) -> None:
    payload = certificate(246, standard=standard)
    if standard:
        payload["program"]["rhs"][-1] = {"num": 2, "den": 1}
    else:
        payload["program"]["constraints"][-1]["rhs"] = {"num": 2, "den": 1}
    result = check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    )
    assert result.objectives_equal
    assert not result.primal_feasible
    assert not result.is_optimal
    payload = certificate(246, standard=standard)
    payload["constraint_dual"][-1] = {"num": 1, "den": 1}
    result = check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    )
    assert not result.dual_feasible
    assert "stationarity" in result.failed_conditions


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.parametrize("n,m", [(128, 0), (1, 1024), (15, 246)])
def test_shared_carrier_does_not_expand_solver_shortcuts(
    standard: bool, n: int, m: int
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.optimization import general_linear_program, linear_program

    candidate = RationalLinearOptimalityCandidate.model_validate(
        zero_certificate(n, m, standard=standard)
    )
    assert check_linear_optimality(candidate).is_optimal
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        from jacobian.math.optimization._models import StandardFormRationalLinearProgram

        if isinstance(candidate.program, StandardFormRationalLinearProgram):
            linear_program(candidate.program)
        else:
            general_linear_program(candidate.program)
    assert (
        exc_info.value.errors()[0]["type"] == "optimization.linear.solver_shape_bound"
    )


@pytest.mark.parametrize("standard", [False, True])
def test_exact_checker_work_boundary(standard: bool) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    accepted = RationalLinearOptimalityCandidate.model_validate(
        zero_certificate(124, 99, standard=standard)
    )
    assert check_linear_optimality(accepted).is_optimal  # 8*125*100 = 100,000
    refused = RationalLinearOptimalityCandidate.model_validate(
        zero_certificate(124, 100, standard=standard)
    )
    with pytest.raises(OperationResourceAdmissionError, match="101000 scalar updates"):
        check_linear_optimality(refused)


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.parametrize("n,m", [(129, 0), (1, 1025), (128, 129)])
def test_carrier_shape_and_cell_limits(standard: bool, n: int, m: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="bound"):
        RationalLinearOptimalityCandidate.model_validate(
            zero_certificate(n, m, standard=standard)
        )


@pytest.mark.parametrize("standard", [False, True])
def test_carrier_cell_boundary_is_structural(standard: bool) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    candidate = RationalLinearOptimalityCandidate.model_validate(
        zero_certificate(128, 128, standard=standard)
    )
    assert (
        RationalLinearOptimalityCandidate.model_validate_json(
            candidate.model_dump_json()
        )
        == candidate
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="candidate check predicts"
    ):
        check_linear_optimality(candidate)


@pytest.mark.parametrize("guard", ["growth", "retention"])
def test_arithmetic_limits_remain_operational(guard: str) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    payload = (
        zero_certificate(124, 99) if guard == "retention" else zero_certificate(1, 0)
    )
    payload["primal_candidate"][0] = (
        {"num": 10**999, "den": 1}
        if guard == "retention"
        else {"num": 1, "den": 10**17000 + 1}
    )
    candidate = RationalLinearOptimalityCandidate.model_validate(payload)
    with pytest.raises(
        OperationResourceAdmissionError, match="candidate check predicts"
    ):
        check_linear_optimality(candidate)


def test_large_candidate_still_requires_exact_source_axes() -> None:
    from pydantic import ValidationError

    payload = certificate(246)
    payload["constraint_dual"].pop()
    with pytest.raises(ValidationError, match="source variable and constraint axes"):
        RationalLinearOptimalityCandidate.model_validate(payload)


def test_prebuilt_constraint_rows_pay_the_same_cell_bound() -> None:
    from pydantic import ValidationError

    from jacobian.math.optimization._general_models import (
        GeneralFormRationalLinearProgram,
        RationalLinearConstraint,
    )

    payload = zero_certificate(128, 129)["program"]
    payload["constraints"] = tuple(
        RationalLinearConstraint.model_validate(row) for row in payload["constraints"]
    )
    with pytest.raises(ValidationError) as exc_info:
        GeneralFormRationalLinearProgram.model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"] == "general_linear_program.raw_input_bound"
    )


def test_large_result_requires_structural_axes_without_rechecking_claim() -> None:
    from pydantic import ValidationError

    candidate = RationalLinearOptimalityCandidate.model_validate(certificate(246))
    result = check_linear_optimality(candidate)
    payload = result.model_dump()
    payload["primal_residuals"] = payload["primal_residuals"][:-1]
    with pytest.raises(ValidationError, match="retained source axes"):
        RationalLinearOptimalityResult.model_validate(payload)
    payload = result.model_dump()
    payload["is_optimal"] = False
    assert not RationalLinearOptimalityResult.model_validate(payload).is_optimal


def test_candidate_and_result_schemas_share_carrier_axes() -> None:
    schema = RationalLinearOptimalityCandidate.model_json_schema()
    assert schema["properties"]["constraint_dual"]["maxItems"] == 1024
    assert schema["properties"]["primal_candidate"]["maxItems"] == 128
    result_schema = RationalLinearOptimalityResult.model_json_schema()
    assert result_schema["properties"]["primal_residuals"]["maxItems"] == 1024
    assert result_schema["properties"]["stationarity_residuals"]["maxItems"] == 128


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.parametrize("field", ["variables", "rows", "coefficients", "candidate"])
def test_native_iterators_do_not_escape_bounded_raw_preflight(
    standard: bool, field: str
) -> None:
    from pydantic import ValidationError

    payload = zero_certificate(128, 129, standard=standard)
    program = payload["program"]
    if field == "variables":
        program["variables"] = iter(program["variables"])
    elif field == "rows":
        key = "coefficients" if standard else "constraints"
        program[key] = iter(program[key])
    elif field == "coefficients":
        # Keep the aggregate count otherwise admissible, but hide one row's
        # size behind an iterator. Reject before consuming its values.
        payload = zero_certificate(1, 1, standard=standard)
        if standard:
            payload["program"]["coefficients"][0] = iter([{"num": 0, "den": 1}])
        else:
            payload["program"]["constraints"][0]["coefficients"] = iter(
                [{"num": 0, "den": 1}]
            )
    else:
        payload["primal_candidate"] = iter(payload["primal_candidate"])
    with pytest.raises(ValidationError, match="bounded list or tuple"):
        RationalLinearOptimalityCandidate.model_validate(payload)


def test_native_general_objective_with_default_empty_rows_composes() -> None:
    from jacobian.math.optimization._general_models import (
        GeneralFormRationalLinearProgram,
        RationalLinearObjective,
    )

    payload = zero_certificate(128, 0)
    program = payload["program"]
    program["objective"] = RationalLinearObjective.model_validate(program["objective"])
    del program["constraints"]
    assert (
        len(GeneralFormRationalLinearProgram.model_validate(program).constraints) == 0
    )
    candidate = RationalLinearOptimalityCandidate.model_validate(payload)
    assert check_linear_optimality(candidate).is_optimal
