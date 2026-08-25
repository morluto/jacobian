"""Source-coordinate integration evidence for general exact rational LPs."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import nextprime
from tests.integration.linear._support import linear_validation_error
from tests.support.rationals import rational_payload as q

from jacobian.math.optimization._general_models import (
    GeneralRationalLinearProgramRequest,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._tools import TOOLS

pytestmark = pytest.mark.requires_backend("flint")


def _program(
    *,
    variables: list[dict[str, object]],
    sense: str,
    objective: list[dict[str, str]],
    constraints: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "variables": variables,
        "objective": {"sense": sense, "coefficients": objective},
        "constraints": constraints,
    }


def _variable(
    name: str,
    lower: dict[str, str] | None = None,
    upper: dict[str, str] | None = None,
) -> dict[str, object]:
    return {"name": name, "lower_bound": lower, "upper_bound": upper}


def _row(
    label: str,
    coefficients: list[dict[str, str]],
    relation: str,
    rhs: dict[str, str],
) -> dict[str, object]:
    return {
        "label": label,
        "coefficients": coefficients,
        "relation": relation,
        "rhs": rhs,
    }


def _run(program: dict[str, object]) -> GeneralRationalLinearProgramResult:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "optimization.linear.rational_general_optimum.compute"
    )
    result = operation.run(
        GeneralRationalLinearProgramRequest.model_validate({"program": program})
    )
    assert isinstance(result, GeneralRationalLinearProgramResult)
    return result


def _fractions(values: object) -> tuple[Fraction, ...]:
    assert values is not None
    return tuple(value.as_fraction() for value in values)


def test_general_lp_replays_le_ge_and_equality_in_original_coordinates() -> None:
    result = _run(
        _program(
            variables=[_variable("x", q(0)), _variable("y", q(0))],
            sense="MINIMIZE",
            objective=[q(1), q(1)],
            constraints=[
                _row("x_cap", [q(1), q(0)], "LE", q(2)),
                _row("y_floor", [q(0), q(1)], "GE", q(1)),
                _row("sum", [q(1), q(1)], "EQ", q(3)),
            ],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (Fraction(2), Fraction(1))
    assert _fractions(result.primal_residuals) == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == 3
    assert _fractions(result.stationarity_residuals) == (Fraction(), Fraction())
    assert result.model_validate_json(result.model_dump_json()) == result


def test_general_lp_preserves_inactive_inequality_residuals_and_lower_shift() -> None:
    result = _run(
        _program(
            variables=[_variable("x", q(3, 2))],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=[_row("inactive", [q(1)], "LE", q(5))],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (Fraction(3, 2),)
    assert _fractions(result.primal_residuals) == (Fraction(-7, 2),)
    assert _fractions(result.constraint_slacks) == (Fraction(7, 2),)
    assert _fractions(result.lower_bound_slacks) == (Fraction(),)
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(3, 2)


@pytest.mark.parametrize(
    ("variable", "objective", "expected"),
    [
        (_variable("x", q(2)), [q(1)], Fraction(2)),
        (_variable("x", None, q(2)), [q(-1)], Fraction(2)),
        (_variable("x", q(-1), q(2)), [q(-1)], Fraction(2)),
        (_variable("x", q(2), q(2)), [q(7)], Fraction(2)),
    ],
)
def test_general_lp_maps_all_closed_bound_shapes(
    variable: dict[str, object],
    objective: list[dict[str, str]],
    expected: Fraction,
) -> None:
    result = _run(
        _program(
            variables=[variable],
            sense="MINIMIZE",
            objective=objective,
            constraints=[],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (expected,)
    assert _fractions(result.stationarity_residuals) == (Fraction(),)


def test_general_lp_maps_a_negative_free_optimum_and_maximization_dual() -> None:
    free_result = _run(
        _program(
            variables=[_variable("x")],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=[_row("fix", [q(1)], "EQ", q(-2))],
        )
    )
    maximum_result = _run(
        _program(
            variables=[_variable("x", q(0))],
            sense="MAXIMIZE",
            objective=[q(1)],
            constraints=[_row("cap", [q(1)], "LE", q(2))],
        )
    )

    assert free_result.status == "OPTIMAL"
    assert _fractions(free_result.primal_candidate) == (Fraction(-2),)
    assert maximum_result.status == "OPTIMAL"
    assert _fractions(maximum_result.primal_candidate) == (Fraction(2),)
    assert _fractions(maximum_result.constraint_dual) == (Fraction(1),)
    assert maximum_result.dual_objective is not None
    assert maximum_result.dual_objective.as_fraction() == 2


def test_general_lp_replays_source_farkas_and_free_unbounded_ray() -> None:
    infeasible = _run(
        _program(
            variables=[_variable("x", q(0))],
            sense="MINIMIZE",
            objective=[q(0)],
            constraints=[_row("contradiction", [q(1)], "LE", q(-1))],
        )
    )
    unbounded = _run(
        _program(
            variables=[_variable("x")],
            sense="MINIMIZE",
            objective=[q(-1)],
            constraints=[],
        )
    )

    assert infeasible.status == "INFEASIBLE"
    assert _fractions(infeasible.farkas_constraints) == (Fraction(1),)
    assert _fractions(infeasible.farkas_lower_bounds) == (Fraction(-1),)
    forged_farkas = deepcopy(infeasible.model_dump(mode="json"))
    forged_farkas["farkas_constraints"] = [q(0)]
    with linear_validation_error():
        GeneralRationalLinearProgramResult.model_validate(forged_farkas)
    assert unbounded.status == "UNBOUNDED"
    assert _fractions(unbounded.recession_direction) == (Fraction(1),)


def test_general_lp_rejects_invalid_bound_order_and_replays_mutation_against_source() -> (
    None
):
    invalid = _program(
        variables=[_variable("x", q(2), q(1))],
        sense="MINIMIZE",
        objective=[q(1)],
        constraints=[],
    )
    with pytest.raises(ValidationError) as caught:
        GeneralRationalLinearProgramRequest.model_validate({"program": invalid})
    assert caught.value.errors()[0]["type"] == "general_linear_program.bound_order"

    result = _run(
        _program(
            variables=[_variable("x", q(0))],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=[_row("minimum", [q(1)], "GE", q(1))],
        )
    )
    assert result.status == "OPTIMAL"
    forged = deepcopy(result.model_dump(mode="json"))
    assert isinstance(forged["program"], dict)
    assert isinstance(forged["program"]["constraints"], list)
    forged["program"]["constraints"][0]["rhs"] = q(2)
    with linear_validation_error():
        GeneralRationalLinearProgramResult.model_validate(forged)
    with linear_validation_error():
        GeneralRationalLinearProgramResult.model_validate(
            {
                "program": result.program,
                "status": "UNKNOWN",
                "primal_candidate": [q(1)],
            }
        )


def test_general_lp_model_validation_uses_owner_codes_for_source_shape_errors() -> None:
    invalid_name = _program(
        variables=[_variable("x-y")],
        sense="MINIMIZE",
        objective=[q(1)],
        constraints=[],
    )
    with pytest.raises(ValidationError) as caught:
        GeneralRationalLinearProgramRequest.model_validate({"program": invalid_name})
    assert (
        caught.value.errors()[0]["type"] == "general_linear_program.variable_identifier"
    )

    invalid_objective = _program(
        variables=[_variable("x")],
        sense="MINIMIZE",
        objective=[q(1), q(2)],
        constraints=[],
    )
    with pytest.raises(ValidationError) as caught:
        GeneralRationalLinearProgramRequest.model_validate(
            {"program": invalid_objective}
        )
    assert caught.value.errors()[0]["type"] == "general_linear_program.objective_length"


def test_general_lp_raw_limits_precede_nested_rational_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian._exact as exact

    def fail_if_called(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("general LP raw admission must precede integer parsing")

    monkeypatch.setattr(exact, "parse_canonical_integer", fail_if_called)
    oversized = "9" * 129
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=[_variable("x", {"num": oversized, "den": "1"})],
                    sense="MINIMIZE",
                    objective=[q(1)],
                    constraints=[],
                )
            }
        )


def test_general_lp_preflights_private_sign_split_and_slack_expansion() -> None:
    variables = [_variable(f"x{index}") for index in range(16)]
    objective = [q(0) for _ in variables]
    constraints = [_row("one_more_slack", [q(0) for _ in variables], "LE", q(0))]
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=variables,
                    sense="MINIMIZE",
                    objective=objective,
                    constraints=constraints,
                )
            }
        )


@pytest.mark.parametrize(
    ("coefficients", "lower", "expected_point", "expected_objective"),
    [
        (q(1), q(10000000), Fraction(10000000), Fraction(10000000)),
        (
            q(1),
            q(12345678901, 97531),
            Fraction(12345678901, 97531),
            Fraction(12345678901, 97531),
        ),
        (
            q(3),
            q(-98765432109, 97531),
            Fraction(-98765432109, 97531),
            Fraction(-296296296327, 97531),
        ),
    ],
)
def test_general_lp_returns_offset_shifted_optima_within_the_mapped_height_bound(
    coefficients: dict[str, str],
    lower: dict[str, str],
    expected_point: Fraction,
    expected_objective: Fraction,
) -> None:
    result = _run(
        _program(
            variables=[_variable("x", lower)],
            sense="MINIMIZE",
            objective=[coefficients],
            constraints=[],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (expected_point,)
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == expected_objective
    assert result.dual_objective is not None
    assert result.dual_objective.as_fraction() == expected_objective
    assert _fractions(result.stationarity_residuals) == (Fraction(),)


def test_general_lp_rejects_source_values_taller_than_the_mapped_result_bound() -> None:
    result = _run(
        _program(
            variables=[_variable("x", q(10000000))],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=[],
        )
    )
    forged = deepcopy(result.model_dump(mode="json"))
    forged["primal_candidate"] = [{"num": "9" * 400, "den": "1"}]
    with linear_validation_error():
        GeneralRationalLinearProgramResult.model_validate(forged)


def test_general_lp_admits_the_full_one_sided_variable_envelope() -> None:
    variables = [_variable(f"x{index}", q(index)) for index in range(32)]
    result = _run(
        _program(
            variables=variables,
            sense="MINIMIZE",
            objective=[q(1) for _ in variables],
            constraints=[],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == tuple(
        Fraction(index) for index in range(32)
    )
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(496)


def test_general_lp_rejects_variables_beyond_the_public_envelope() -> None:
    variables = [_variable(f"x{index}", q(0)) for index in range(33)]
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=variables,
                    sense="MINIMIZE",
                    objective=[q(1) for _ in variables],
                    constraints=[],
                )
            }
        )


def test_general_lp_defers_free_split_admission_to_the_normalized_columns() -> None:
    variables = [_variable(f"x{index}") for index in range(17)]
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=variables,
                    sense="MINIMIZE",
                    objective=[q(1) for _ in variables],
                    constraints=[],
                )
            }
        )


def test_general_lp_admits_sixty_four_trivial_equalities_within_the_row_envelope() -> (
    None
):
    constraints = [_row(f"trivial_{index}", [q(0)], "EQ", q(0)) for index in range(64)]
    result = _run(
        _program(
            variables=[_variable("x", q(0))],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=constraints,
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (Fraction(),)
    assert _fractions(result.primal_residuals) == (Fraction(),) * 64
    assert result.model_validate_json(result.model_dump_json()) == result


def test_general_lp_rejects_constraints_beyond_the_public_row_envelope() -> None:
    constraints = [_row(f"row_{index}", [q(1)], "EQ", q(0)) for index in range(65)]
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=[_variable("x", q(0))],
                    sense="MINIMIZE",
                    objective=[q(1)],
                    constraints=constraints,
                )
            }
        )


def test_general_lp_defers_upper_expansion_rows_to_normalized_admission() -> None:
    constraints = [_row(f"row_{index}", [q(1)], "EQ", q(0)) for index in range(64)]
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=[_variable("x", q(0), q(1))],
                    sense="MINIMIZE",
                    objective=[q(1)],
                    constraints=constraints,
                )
            }
        )


def test_general_lp_admits_generated_intermediates_from_admitted_source_heights() -> (
    None
):
    scale = 10**127
    result = _run(
        _program(
            variables=[_variable("x", q(scale))],
            sense="MINIMIZE",
            objective=[q(1)],
            constraints=[_row("scaled_floor", [q(scale)], "GE", q(0))],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == (Fraction(scale),)
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(scale)
    assert _fractions(result.primal_residuals) == (Fraction(scale**2),)
    assert result.dual_objective is not None
    assert result.dual_objective.as_fraction() == Fraction(scale)


def test_general_lp_still_rejects_expansions_beyond_the_derived_envelope() -> None:
    prime = 10**127 + 283
    constraints = []
    for row_index in range(10):
        coefficients = []
        for _ in range(16):
            prime = nextprime(prime)
            coefficients.append(q(1, prime))
        constraints.append(_row(f"dense_{row_index}", coefficients, "EQ", q(0)))
    with linear_validation_error():
        GeneralRationalLinearProgramRequest.model_validate(
            {
                "program": _program(
                    variables=[_variable(f"x{index}") for index in range(16)],
                    sense="MINIMIZE",
                    objective=[q(1)] * 16,
                    constraints=constraints,
                )
            }
        )


_TALL_PRIMES = (
    10**127 + 283,
    10**127 + 1521,
    10**127 + 1533,
)


def test_general_lp_keeps_multi_offset_optima_within_the_mapped_bounds() -> None:
    lowers = tuple(Fraction(1, prime) for prime in _TALL_PRIMES)
    result = _run(
        _program(
            variables=[
                _variable(f"x{index}", q(1, prime))
                for index, prime in enumerate(_TALL_PRIMES)
            ],
            sense="MINIMIZE",
            objective=[q(1)] * len(_TALL_PRIMES),
            constraints=[],
        )
    )

    assert result.status == "OPTIMAL"
    assert _fractions(result.primal_candidate) == lowers
    expected = sum(lowers)
    assert len(str(expected.denominator)) > 300
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == expected
    assert result.dual_objective is not None
    assert result.dual_objective.as_fraction() == expected
    assert result.model_validate_json(result.model_dump_json()) == result
