"""Independent primal-dual conditions and checking-versus-search boundaries."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization import check_linear_optimality, general_linear_program
from jacobian.math.optimization._general_models import GeneralFormRationalLinearProgram
from jacobian.math.optimization._optimality import (
    RationalLinearOptimalityCandidate,
    RationalLinearOptimalityResult,
)
from jacobian.math.optimization._tools import TOOLS


def _r(value: int) -> dict[str, int]:
    return {"num": value, "den": 1}


def _candidate(
    *, sense: str = "MINIMIZE", x: int = 1, y: int = 1, free: bool = False
) -> RationalLinearOptimalityCandidate:
    return RationalLinearOptimalityCandidate.model_validate(
        {
            "program": {
                "variables": [
                    {
                        "name": "x",
                        "lower_bound": None if free else _r(0),
                        "upper_bound": None,
                    }
                ],
                "objective": {"sense": sense, "coefficients": [_r(1)]},
                "constraints": [
                    {
                        "label": "row",
                        "coefficients": [_r(1)],
                        "relation": "EQ"
                        if free
                        else "GE"
                        if sense == "MINIMIZE"
                        else "LE",
                        "rhs": _r(1),
                    }
                ],
            },
            "primal_candidate": [_r(x)],
            "constraint_dual": [_r(y)],
            "lower_bound_dual": [_r(0)],
            "upper_bound_dual": [_r(0)],
        }
    )


@pytest.mark.parametrize("sense", ["MINIMIZE", "MAXIMIZE"])
@pytest.mark.parametrize("free", [False, True])
def test_valid_pairs_follow_source_signs_and_survive_serialization(
    sense: str, free: bool
) -> None:
    candidate = _candidate(sense=sense, free=free)
    result = check_linear_optimality(candidate)
    assert result.is_optimal
    assert result.primal_objective.as_fraction() == Fraction(1)
    assert result.failed_conditions == ()
    decoded = RationalLinearOptimalityResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert check_linear_optimality(decoded.candidate).is_optimal


@pytest.mark.parametrize(
    "x,y,failed",
    [
        (0, 1, "primal_constraints"),
        (1, 2, "stationarity"),
        (2, 1, "objective_equality"),
        (0, 0, "primal_constraints"),
    ],
)
def test_perturbed_candidates_do_not_establish_optimality(
    x: int, y: int, failed: str
) -> None:
    result = check_linear_optimality(_candidate(x=x, y=y))
    assert not result.is_optimal
    assert failed in result.failed_conditions
    if x == y == 0:
        assert result.objectives_equal
        assert not result.primal_feasible


@pytest.mark.parametrize(
    "sense,bound,multiplier",
    [("MINIMIZE", "lower_bound", 1), ("MAXIMIZE", "upper_bound", 1)],
)
def test_active_bounds_have_source_multipliers(
    sense: str, bound: str, multiplier: int
) -> None:
    payload = _candidate(sense=sense).model_dump()
    payload["program"]["constraints"] = []
    payload["program"]["variables"][0][bound] = _r(1)
    payload["constraint_dual"] = []
    payload["lower_bound_dual" if bound == "lower_bound" else "upper_bound_dual"] = [
        _r(multiplier)
    ]
    result = check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    )
    assert result.is_optimal


def test_absent_bound_multiplier_cannot_fake_stationarity() -> None:
    payload = _candidate(free=True, y=0).model_dump()
    payload["lower_bound_dual"] = [_r(1)]
    result = check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    )
    assert not result.dual_feasible
    assert "dual_bound_signs" in result.failed_conditions


def test_wrong_dual_sign_is_rejected_even_with_equal_objectives() -> None:
    payload = _candidate(x=0, y=-1).model_dump()
    payload["program"]["constraints"][0]["rhs"] = _r(0)
    payload["program"]["objective"]["coefficients"] = [_r(-1)]
    result = check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    )
    assert result.primal_feasible and result.objectives_equal
    assert not result.dual_feasible
    assert "dual_constraint_signs" in result.failed_conditions


def test_checker_accepts_a_certificate_beyond_normalized_solver_envelope() -> None:
    n = 32
    candidate = RationalLinearOptimalityCandidate.model_validate(
        {
            "program": {
                "variables": [{"name": f"x{i}"} for i in range(n)],
                "objective": {"sense": "MINIMIZE", "coefficients": [_r(1)] * n},
                "constraints": [
                    {
                        "label": f"row{i}",
                        "coefficients": [_r(int(i == j)) for j in range(n)],
                        "relation": "EQ",
                        "rhs": _r(1),
                    }
                    for i in range(n)
                ],
            },
            "primal_candidate": [_r(1)] * n,
            "constraint_dual": [_r(1)] * n,
            "lower_bound_dual": [_r(0)] * n,
            "upper_bound_dual": [_r(0)] * n,
        }
    )
    with pytest.raises(OperationResourceAdmissionError):
        assert isinstance(candidate.program, GeneralFormRationalLinearProgram)
        general_linear_program(candidate.program)
    result = check_linear_optimality(candidate)
    assert result.is_optimal
    assert result.primal_objective.as_fraction() == n


def test_standard_form_public_example_checks_without_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.optimization.operations as solver

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("candidate checking must not invoke optimization")

    monkeypatch.setattr(solver, "linear_program", unexpected)
    operation = next(
        tool for tool in TOOLS if tool.operation_id.endswith("optimality.check")
    )
    request = operation.request_type.model_validate_json(
        json.dumps(operation.examples[0].input)
    )
    result = operation.run(request)
    assert result.is_optimal
    assert type(result.candidate) is RationalLinearOptimalityCandidate
    decoded = RationalLinearOptimalityResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert check_linear_optimality(decoded.candidate).is_optimal


def test_malformed_axes_and_ingestion_are_not_negative_verdicts() -> None:
    payload = _candidate().model_dump()
    payload["primal_candidate"] = []
    with pytest.raises(ValidationError, match="source"):
        RationalLinearOptimalityCandidate.model_validate(payload)


def test_large_shared_denominators_do_not_pay_independent_growth() -> None:
    payload = _candidate().model_dump()
    payload["program"]["objective"]["coefficients"] = [{"num": 1, "den": 10**127 + 1}]
    payload["constraint_dual"] = [{"num": 1, "den": 10**127 + 1}]
    assert check_linear_optimality(
        RationalLinearOptimalityCandidate.model_validate(payload)
    ).is_optimal
    payload["primal_candidate"] = [
        {"num": "1" * (MAX_CANONICAL_RATIONAL_DIGITS + 1), "den": "1"}
    ]
    with pytest.raises(ValidationError, match=f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit"):
        RationalLinearOptimalityCandidate.model_validate(payload)


@pytest.mark.parametrize("serialized", [False, True])
def test_solver_candidates_above_source_digit_limit_compose(serialized: bool) -> None:
    coefficient = 10**100
    program = GeneralFormRationalLinearProgram.model_validate(
        {
            "variables": [
                {"name": name, "lower_bound": None, "upper_bound": None}
                for name in ("x", "y")
            ],
            "objective": {"sense": "MINIMIZE", "coefficients": [_r(0), _r(0)]},
            "constraints": [
                {
                    "label": "first",
                    "coefficients": [_r(coefficient), _r(1)],
                    "relation": "EQ",
                    "rhs": _r(0),
                },
                {
                    "label": "second",
                    "coefficients": [_r(0), _r(coefficient)],
                    "relation": "EQ",
                    "rhs": _r(1),
                },
            ],
        }
    )
    solved = general_linear_program(program)
    assert solved.primal_candidate is not None
    assert solved.primal_candidate[0].as_fraction() == Fraction(-1, coefficient**2)
    fields = (
        "program",
        "primal_candidate",
        "constraint_dual",
        "lower_bound_dual",
        "upper_bound_dual",
    )
    payload = (
        {name: solved.model_dump(mode="json")[name] for name in fields}
        if serialized
        else {name: getattr(solved, name) for name in fields}
    )
    candidate = (
        RationalLinearOptimalityCandidate.model_validate_json(json.dumps(payload))
        if serialized
        else RationalLinearOptimalityCandidate.model_validate(payload)
    )
    checked = check_linear_optimality(candidate)
    assert checked.is_optimal
    assert checked.stationarity_residuals == solved.stationarity_residuals
    assert (
        RationalLinearOptimalityResult.model_validate_json(checked.model_dump_json())
        == checked
    )


@pytest.mark.parametrize("shared", [False, True])
def test_gap_admission_accounts_for_all_primal_and_dual_products(shared: bool) -> None:
    n, m = 32, 33
    offset = 0

    def scalar() -> dict[str, int]:
        nonlocal offset
        offset += 1
        return {"num": 1, "den": 10**127 + (1 if shared else offset)}

    candidate = RationalLinearOptimalityCandidate.model_validate(
        {
            "program": {
                "variables": [
                    {
                        "name": f"x{i}",
                        "lower_bound": scalar(),
                        "upper_bound": {"num": 1, "den": 1},
                    }
                    for i in range(n)
                ],
                "objective": {
                    "sense": "MINIMIZE",
                    "coefficients": [scalar() for _ in range(n)],
                },
                "constraints": [
                    {
                        "label": f"row{i}",
                        "coefficients": [scalar() for _ in range(n)],
                        "relation": "EQ",
                        "rhs": scalar(),
                    }
                    for i in range(m)
                ],
            },
            "primal_candidate": [scalar() for _ in range(n)],
            "constraint_dual": [scalar() for _ in range(m)],
            "lower_bound_dual": [scalar() for _ in range(n)],
            "upper_bound_dual": [scalar() for _ in range(n)],
        }
    )
    if shared:
        assert not check_linear_optimality(candidate).is_optimal
    else:
        with pytest.raises(OperationResourceAdmissionError, match="rational digits"):
            check_linear_optimality(candidate)
