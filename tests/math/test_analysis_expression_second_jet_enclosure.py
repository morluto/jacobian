from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.analysis._models import (
    DyadicClosedInterval,
    IntervalExpressionSecondJetEnclosureRequest,
    IntervalExpressionSecondJetEnclosureResult,
)
from jacobian.math.analysis._operations import _second_jet_enclosure


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    value = Fraction(numerator, denominator)
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _var(name: str) -> dict[str, Any]:
    return {"op": "var", "variable": name}


def _const(value: int) -> dict[str, Any]:
    return {"op": "const", "value": _q(value)}


def _request(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    *,
    precision_bits: int = 128,
) -> IntervalExpressionSecondJetEnclosureRequest:
    return IntervalExpressionSecondJetEnclosureRequest.model_validate(
        {
            "expression": expression,
            "box": {
                "variables": [name for name, _, _ in coordinates],
                "intervals": [
                    {
                        "lower": _q(lower.numerator, lower.denominator),
                        "upper": _q(upper.numerator, upper.denominator),
                    }
                    for _, lower, upper in coordinates
                ],
            },
            "precision_bits": precision_bits,
        }
    )


def _run(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
) -> IntervalExpressionSecondJetEnclosureResult:
    return _second_jet_enclosure(_request(expression, coordinates))


def _contains(interval: DyadicClosedInterval, value: Fraction) -> bool:
    return interval.lower.as_fraction() <= value <= interval.upper.as_fraction()


def _gradient(
    result: IntervalExpressionSecondJetEnclosureResult,
) -> dict[str, DyadicClosedInterval]:
    return {entry.variable: entry.enclosure for entry in result.gradient}


def _hessian(
    result: IntervalExpressionSecondJetEnclosureResult,
) -> dict[tuple[str, str], DyadicClosedInterval]:
    return {
        (entry.first_variable, entry.second_variable): entry.enclosure
        for entry in result.hessian
    }


def test_quadratic_encloses_known_gradient_and_hessian() -> None:
    result = _run(
        {
            "op": "add",
            "children": [
                {"op": "pow", "exponent": 2, "children": [_var("x")]},
                {"op": "pow", "exponent": 2, "children": [_var("y")]},
            ],
        },
        (("x", Fraction(-1), Fraction(2)), ("y", Fraction(3), Fraction(4))),
    )

    assert result.status == "ENCLOSED"
    assert result.value is not None
    assert _contains(result.value, Fraction(10))
    gradient = _gradient(result)
    assert _contains(gradient["x"], Fraction(-2))
    assert _contains(gradient["x"], Fraction(4))
    assert _contains(gradient["y"], Fraction(6))
    assert _contains(gradient["y"], Fraction(8))
    hessian = _hessian(result)
    assert _contains(hessian[("x", "x")], Fraction(2))
    assert _contains(hessian[("x", "y")], Fraction(0))
    assert _contains(hessian[("y", "y")], Fraction(2))


def test_product_has_constant_mixed_second_partial() -> None:
    result = _run(
        {"op": "mul", "children": [_var("x"), _var("y")]},
        (("x", Fraction(2), Fraction(3)), ("y", Fraction(5), Fraction(7))),
    )

    assert result.status == "ENCLOSED"
    hessian = _hessian(result)
    assert _contains(hessian[("x", "x")], Fraction(0))
    assert _contains(hessian[("x", "y")], Fraction(1))
    assert _contains(hessian[("y", "y")], Fraction(0))


@pytest.mark.parametrize(
    ("expression", "coordinates", "gradient", "hessian"),
    [
        (
            {
                "op": "exp",
                "children": [{"op": "mul", "children": [_var("x"), _var("y")]}],
            },
            (("x", Fraction(0), Fraction(0)), ("y", Fraction(0), Fraction(0))),
            {"x": Fraction(0), "y": Fraction(0)},
            {("x", "x"): Fraction(0), ("x", "y"): Fraction(1), ("y", "y"): Fraction(0)},
        ),
        (
            {
                "op": "log",
                "children": [{"op": "add", "children": [_var("x"), _var("y")]}],
            },
            (("x", Fraction(1), Fraction(1)), ("y", Fraction(1), Fraction(1))),
            {"x": Fraction(1, 2), "y": Fraction(1, 2)},
            {
                ("x", "x"): Fraction(-1, 4),
                ("x", "y"): Fraction(-1, 4),
                ("y", "y"): Fraction(-1, 4),
            },
        ),
        (
            {"op": "sqrt", "children": [_var("x")]},
            (("x", Fraction(4), Fraction(4)),),
            {"x": Fraction(1, 4)},
            {("x", "x"): Fraction(-1, 32)},
        ),
    ],
)
def test_forward_chain_rules_enclose_known_point_derivatives(
    expression: dict[str, Any],
    coordinates: tuple[tuple[str, Fraction, Fraction], ...],
    gradient: dict[str, Fraction],
    hessian: dict[tuple[str, str], Fraction],
) -> None:
    result = _run(expression, coordinates)

    assert result.status == "ENCLOSED"
    actual_gradient = _gradient(result)
    actual_hessian = _hessian(result)
    for variable, expected in gradient.items():
        assert _contains(actual_gradient[variable], expected)
    for pair, expected in hessian.items():
        assert _contains(actual_hessian[pair], expected)


def test_axis_permutation_transports_gradient_and_upper_triangle() -> None:
    expression = {
        "op": "add",
        "children": [
            {"op": "mul", "children": [_var("x"), _var("y")]},
            {"op": "pow", "exponent": 2, "children": [_var("x")]},
        ],
    }
    xy = _run(
        expression,
        (("x", Fraction(1), Fraction(1)), ("y", Fraction(2), Fraction(2))),
    )
    yx = _run(
        expression,
        (("y", Fraction(2), Fraction(2)), ("x", Fraction(1), Fraction(1))),
    )

    assert tuple(entry.variable for entry in xy.gradient) == ("x", "y")
    assert tuple(entry.variable for entry in yx.gradient) == ("y", "x")
    assert tuple(_hessian(xy)) == (("x", "x"), ("x", "y"), ("y", "y"))
    assert tuple(_hessian(yx)) == (("y", "y"), ("y", "x"), ("x", "x"))
    assert _gradient(xy)["x"] == _gradient(yx)["x"]
    assert _gradient(xy)["y"] == _gradient(yx)["y"]
    assert _hessian(xy)[("x", "y")] == _hessian(yx)[("y", "x")]


def test_dependency_keeps_the_correct_zero_derivatives() -> None:
    result = _run(
        {"op": "sub", "children": [_var("x"), _var("x")]},
        (("x", Fraction(-1), Fraction(1)),),
    )

    assert result.status == "ENCLOSED"
    assert _contains(_gradient(result)["x"], Fraction(0))
    assert _contains(_hessian(result)[("x", "x")], Fraction(0))


def test_constant_expression_retains_the_zero_dimensional_second_jet() -> None:
    result = _run(
        {"op": "pow", "exponent": 3, "children": [_const(2)]},
        (),
    )

    assert result.status == "ENCLOSED"
    assert result.value is not None
    assert _contains(result.value, Fraction(8))
    assert result.gradient == ()
    assert result.hessian == ()


def test_sine_box_crossing_an_interior_extremum_encloses_second_derivative() -> None:
    result = _run(
        {"op": "sin", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(2)),),
    )

    assert result.status == "ENCLOSED"
    assert _contains(_gradient(result)["x"], Fraction(0))
    assert _contains(_hessian(result)[("x", "x")], Fraction(-1))


def test_sqrt_touching_zero_is_a_typed_second_derivative_nonconclusion() -> None:
    result = _run(
        {"op": "sqrt", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )

    assert result.status == "DOMAIN_UNPROVEN"
    assert result.value is None
    assert result.domain_failure is not None
    assert (
        result.domain_failure.reason
        == "SQRT_ARGUMENT_NOT_STRICTLY_POSITIVE_FOR_SECOND_JET"
    )


def test_source_bound_result_round_trips_and_rejects_mutated_partial() -> None:
    result = _run(
        {"op": "div", "children": [_const(1), _var("x")]},
        (("x", Fraction(1), Fraction(2)),),
    )

    payload = result.model_dump(mode="json")
    assert (
        IntervalExpressionSecondJetEnclosureResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    payload = deepcopy(payload)
    payload["hessian"][0]["enclosure"]["lower"] = {"mantissa": "0", "exponent": 0}
    with pytest.raises(ValidationError, match="does not replay"):
        IntervalExpressionSecondJetEnclosureResult.model_validate(payload)


def test_request_strict_json_transport_round_trip() -> None:
    request = _request(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    assert (
        IntervalExpressionSecondJetEnclosureRequest.model_validate_json(
            request.model_dump_json(), strict=True
        )
        == request
    )


def _balanced_binary_tree(op: str, leaf_cycle: tuple[str, ...]) -> dict[str, Any]:
    leaves = tuple(leaf_cycle[index % len(leaf_cycle)] for index in range(32))

    def build(names: tuple[str, ...]) -> dict[str, Any]:
        if len(names) == 1:
            return _var(names[0])
        middle = len(names) // 2
        return {
            "op": op,
            "children": [build(names[:middle]), build(names[middle:])],
        }

    return build(leaves)


def test_four_variable_affine_jet_fits_its_dimension_derived_budget() -> None:
    expression: dict[str, Any] = _var("a")
    for variable in ("b", "c", "d"):
        expression = {"op": "add", "children": [expression, _var(variable)]}
    result = _run(
        expression,
        tuple(
            (variable, Fraction(0), Fraction(1)) for variable in ("a", "b", "c", "d")
        ),
    )

    assert result.status == "ENCLOSED"
    assert result.value is not None
    assert _contains(result.value, Fraction(4))
    gradient = _gradient(result)
    assert tuple(gradient) == ("a", "b", "c", "d")
    for variable in ("a", "b", "c", "d"):
        assert _contains(gradient[variable], Fraction(1))
    hessian = _hessian(result)
    assert len(hessian) == 10
    for pair in hessian:
        assert _contains(hessian[pair], Fraction(0))


def test_full_box_affine_jet_encloses_all_eight_variables() -> None:
    variables = tuple("abcdefgh")
    expression: dict[str, Any] = _var(variables[0])
    for variable in variables[1:]:
        expression = {"op": "add", "children": [expression, _var(variable)]}
    result = _run(
        expression,
        tuple((variable, Fraction(0), Fraction(1)) for variable in variables),
    )

    assert result.status == "ENCLOSED"
    assert result.value is not None
    assert _contains(result.value, Fraction(4))
    gradient = _gradient(result)
    assert tuple(gradient) == variables
    hessian = _hessian(result)
    assert len(hessian) == 36


def test_work_budget_scales_with_the_jet_dimension() -> None:
    box = tuple((variable, Fraction(2), Fraction(3)) for variable in "abcdefgh")
    wide_tree = _balanced_binary_tree("div", tuple("abcdefgh"))
    with pytest.raises(ValidationError, match="exceeds its"):
        _request(wide_tree, box)

    narrow_box = (
        ("a", Fraction(2), Fraction(3)),
        ("b", Fraction(2), Fraction(3)),
        ("c", Fraction(2), Fraction(3)),
    )
    same_shaped_tree = _balanced_binary_tree("mul", ("a", "b", "c"))
    assert _run(same_shaped_tree, narrow_box).status == "ENCLOSED"


def test_backend_failure_returns_a_typed_nonconclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.analysis._operations as operations

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic backend rejection")

    monkeypatch.setattr(operations, "_evaluate_second_jet", fail)
    result = _run(
        {"op": "exp", "children": [_var("x")]},
        (("x", Fraction(0), Fraction(1)),),
    )
    assert result.status == "BACKEND_ERROR"
    assert result.value is None
    assert result.gradient == ()
    assert result.hessian == ()
