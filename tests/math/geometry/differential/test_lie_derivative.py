"""Exact contract tests for rational coordinate-tensor Lie derivatives."""

from __future__ import annotations

from itertools import product
from typing import Any

import pytest
import sympy
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential import (
    RationalCoordinateTensor,
    RationalLieDerivativeProfile,
    RationalLieDerivativeRequest,
    lie_derivative,
)
from jacobian.math.geometry.differential import _bounds as lie_bounds
from jacobian.math.geometry.differential import _sympy as lie_backend
from jacobian.math.geometry.differential._bounds import (
    MAX_LIE_DERIVATIVE_WORK_UNITS,
    build_lie_derivative_plan,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_RANK,
)
from jacobian.math.polynomials._conversions import rational_function_to_sympy
from jacobian.math.polynomials.values import RationalFunction

type Coefficient = int | tuple[int, int]
type PolynomialTerm = tuple[Coefficient, tuple[int, ...]]


def _sparse(*terms: PolynomialTerm) -> dict[str, Any]:
    return {
        "terms": [
            {
                "coefficient": {
                    "num": str(
                        coefficient if isinstance(coefficient, int) else coefficient[0]
                    ),
                    "den": str(1 if isinstance(coefficient, int) else coefficient[1]),
                },
                "exponents": list(exponents),
            }
            for coefficient, exponents in terms
        ]
    }


def _function(
    variables: tuple[str, ...],
    *numerator: PolynomialTerm,
    denominator: tuple[PolynomialTerm, ...] | None = None,
) -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "variables": list(variables),
            "numerator": _sparse(*numerator),
            "denominator": _sparse(
                *((1, (0,) * len(variables)),) if denominator is None else denominator
            ),
        }
    )


def _zero(variables: tuple[str, ...]) -> RationalFunction:
    return _function(variables)


def _guard(*terms: PolynomialTerm) -> dict[str, Any]:
    return _sparse(*terms)


def _tensor(
    variables: tuple[str, ...],
    variance: tuple[str, ...],
    components: tuple[RationalFunction, ...],
    *,
    guards: tuple[dict[str, Any], ...] = (),
) -> RationalCoordinateTensor:
    return RationalCoordinateTensor.model_validate(
        {
            "coordinate_axis": list(variables),
            "variance": list(variance),
            "components": [
                component.model_dump(mode="json") for component in components
            ],
            "retained_nonzero_denominators": list(guards),
        }
    )


def _expressions(tensor: RationalCoordinateTensor) -> tuple[Any, ...]:
    return tuple(rational_function_to_sympy(value) for value in tensor.components)


def _assert_expression(actual: RationalFunction, expected: Any) -> None:
    assert sympy.cancel(rational_function_to_sympy(actual) - expected) == 0


def test_rank_zero_scalar_replays_directional_derivative() -> None:
    variables = ("x", "y")
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (0, 1))),
            _function(variables, (1, (1, 0))),
        ),
    )
    scalar = _tensor(
        variables,
        (),
        (_function(variables, (1, (2, 1))),),
    )

    result = lie_derivative(vector, scalar)

    assert result.lie_derivative == _tensor(
        variables,
        (),
        (_function(variables, (1, (3, 0)), (2, (1, 2))),),
    )


def test_fractional_degree_two_inputs_use_content_aware_height_admission() -> None:
    variables = ("x",)
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(
                variables,
                ((1, 3), (2,)),
                ((1, 2), (1,)),
                (1, (0,)),
                denominator=((1, (1,)), (1, (0,))),
            ),
        ),
        guards=(_guard((1, (1,)), (1, (0,))),),
    )
    scalar = _tensor(
        variables,
        (),
        (
            _function(
                variables,
                ((1, 4), (2,)),
                ((1, 3), (1,)),
                ((1, 2), (0,)),
                denominator=((1, (1,)), (2, (0,))),
            ),
        ),
        guards=(_guard((1, (1,)), (2, (0,))),),
    )

    result = lie_derivative(vector, scalar).lie_derivative.components[0]
    x = sympy.Symbol("x")
    expected = (6 * x**4 + 33 * x**3 + 58 * x**2 + 78 * x + 12) / (
        72 * x**3 + 360 * x**2 + 576 * x + 288
    )

    _assert_expression(result, expected)
    assert (
        max(
            len(component.lstrip("-"))
            for polynomial in (result.numerator, result.denominator)
            for term in polynomial.terms
            for component in (term.coefficient.num, term.coefficient.den)
        )
        == 2
    )


def test_content_aware_height_admission_rejects_real_output_growth() -> None:
    variables = ("x",)
    # Each input coefficient has 65 digits; their exact product has 129, one
    # beyond the canonical rational-function carrier.
    large_coefficient = 10**64
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (_function(variables, (large_coefficient, (0,))),),
    )
    scalar = _tensor(
        variables,
        (),
        (_function(variables, (large_coefficient, (1,))),),
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="coefficient bound",
    ) as error:
        build_lie_derivative_plan(vector, scalar)

    assert error.value.errors()[0]["type"].endswith("result_height")


def test_vector_lie_derivative_is_the_antisymmetric_lie_bracket() -> None:
    variables = ("x", "y")
    x_field = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(
                variables,
                (1, (0, 0)),
                denominator=((1, (1, 0)),),
            ),
            _zero(variables),
        ),
        guards=(_guard((1, (1, 0))),),
    )
    y_field = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (2, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )

    xy = lie_derivative(x_field, y_field).lie_derivative
    yx = lie_derivative(y_field, x_field).lie_derivative

    expected = (sympy.Integer(3), sympy.Integer(0))
    for actual, reverse, value in zip(
        xy.components, yx.components, expected, strict=True
    ):
        _assert_expression(actual, value)
        assert (
            sympy.cancel(
                rational_function_to_sympy(actual) + rational_function_to_sympy(reverse)
            )
            == 0
        )

    assert xy.retained_nonzero_denominators == (
        x_field.retained_nonzero_denominators[0],
    )
    assert yx.retained_nonzero_denominators == (
        x_field.retained_nonzero_denominators[0],
    )


def test_constant_vector_and_covector_have_opposite_correction_signs() -> None:
    variables = ("x", "y")
    dilation = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )
    constant_components = (
        _function(variables, (1, (0, 0))),
        _zero(variables),
    )

    vector_result = lie_derivative(
        dilation,
        _tensor(variables, ("CONTRAVARIANT",), constant_components),
    ).lie_derivative
    covector_result = lie_derivative(
        dilation,
        _tensor(variables, ("COVARIANT",), constant_components),
    ).lie_derivative

    assert _expressions(vector_result) == (-1, 0)
    assert _expressions(covector_result) == (1, 0)


def test_rotation_kills_euclidean_metric_and_dilation_scales_it() -> None:
    variables = ("x", "y")
    zero = _zero(variables)
    one = _function(variables, (1, (0, 0)))
    euclidean_metric = _tensor(
        variables,
        ("COVARIANT", "COVARIANT"),
        (one, zero, zero, one),
    )
    rotation = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (-1, (0, 1))),
            _function(variables, (1, (1, 0))),
        ),
    )
    dilation = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )

    assert _expressions(lie_derivative(rotation, euclidean_metric).lie_derivative) == (
        0,
        0,
        0,
        0,
    )
    assert _expressions(lie_derivative(dilation, euclidean_metric).lie_derivative) == (
        2,
        0,
        0,
        2,
    )


def test_cancellation_retains_the_source_nonvanishing_locus_through_composition() -> (
    None
):
    variables = ("x", "y")
    x_guard = _guard((1, (1, 0)))
    rational_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(
                variables,
                (1, (0, 0)),
                denominator=((1, (1, 0)),),
            ),
            _zero(variables),
        ),
        guards=(x_guard,),
    )
    scalar = _tensor(
        variables,
        (),
        (_function(variables, (1, (2, 0))),),
    )

    first = lie_derivative(rational_vector, scalar)

    assert _expressions(first.lie_derivative) == (2,)
    assert first.lie_derivative.retained_nonzero_denominators == (
        rational_vector.retained_nonzero_denominators[0],
    )

    polynomial_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0))),
            _zero(variables),
        ),
    )
    serialized = first.lie_derivative.model_dump(mode="json")
    second = lie_derivative(
        polynomial_vector,
        RationalCoordinateTensor.model_validate(serialized),
    )
    assert _expressions(second.lie_derivative) == (0,)
    assert second.lie_derivative.retained_nonzero_denominators == (
        rational_vector.retained_nonzero_denominators[0],
    )


def test_coordinate_permutation_transports_axis_components_and_values() -> None:
    variables = ("y", "x")
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (0, 1))),
            _function(variables, (1, (1, 0))),
        ),
    )
    scalar = _tensor(
        variables,
        (),
        (_function(variables, (1, (1, 2))),),
    )

    result = lie_derivative(vector, scalar).lie_derivative

    assert result.coordinate_axis == variables
    assert result.components[0] == _function(variables, (2, (2, 1)), (1, (0, 3)))


def test_small_corpus_matches_an_independent_expression_formula() -> None:
    variables = ("x", "y")
    x, y = sympy.symbols("x y")
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0)), (1, (0, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )
    tensors = (
        _tensor(variables, (), (_function(variables, (1, (1, 1))),)),
        _tensor(
            variables,
            ("COVARIANT",),
            (
                _function(variables, (1, (2, 0))),
                _function(variables, (1, (0, 2))),
            ),
        ),
        _tensor(
            variables,
            ("COVARIANT", "COVARIANT"),
            (
                _function(variables, (1, (2, 0))),
                _function(variables, (1, (1, 1))),
                _function(variables, (1, (1, 1))),
                _function(variables, (1, (0, 2)), (1, (0, 0))),
            ),
        ),
    )
    vector_expressions = (x + 1, y)

    for tensor in tensors:
        actual = lie_derivative(vector, tensor).lie_derivative
        source = _expressions(tensor)
        for index in product(range(2), repeat=len(tensor.variance)):
            offset = sum(
                value * 2 ** (len(index) - position - 1)
                for position, value in enumerate(index)
            )
            expected = sum(
                vector_expressions[axis] * sympy.diff(source[offset], (x, y)[axis])
                for axis in range(2)
            )
            for position, variance in enumerate(tensor.variance):
                for axis in range(2):
                    replaced = (*index[:position], axis, *index[position + 1 :])
                    replaced_offset = sum(
                        value * 2 ** (len(replaced) - slot - 1)
                        for slot, value in enumerate(replaced)
                    )
                    if variance == "CONTRAVARIANT":
                        expected -= (
                            sympy.diff(
                                vector_expressions[index[position]], (x, y)[axis]
                            )
                            * source[replaced_offset]
                        )
                    else:
                        expected += (
                            sympy.diff(
                                vector_expressions[axis], (x, y)[index[position]]
                            )
                            * source[replaced_offset]
                        )
            _assert_expression(actual.components[offset], expected)


def test_rejects_axis_mismatch_and_nonvector_signature_at_domain_admission() -> None:
    x_axis = ("x",)
    y_axis = ("y",)
    contravariant_x = _tensor(
        x_axis,
        ("CONTRAVARIANT",),
        (_function(x_axis, (1, (1,))),),
    )
    covariant_x = _tensor(
        x_axis,
        ("COVARIANT",),
        (_function(x_axis, (1, (1,))),),
    )
    scalar_y = _tensor(y_axis, (), (_function(y_axis, (1, (1,))),))
    scalar_x = _tensor(x_axis, (), (_function(x_axis, (1, (1,))),))

    with pytest.raises(
        OperationDomainValidationError,
        match="same ordered coordinate axis",
    ) as mismatch:
        lie_derivative(contravariant_x, scalar_y)
    assert mismatch.value.errors()[0]["type"].endswith("coordinate_axis_mismatch")

    with pytest.raises(
        OperationDomainValidationError,
        match="rank one and CONTRAVARIANT",
    ) as signature:
        lie_derivative(covariant_x, scalar_x)
    assert signature.value.errors()[0]["type"].endswith("vector_signature")


def test_nonvector_signature_precedes_nested_coprimality_recognition() -> None:
    nonreduced = {
        "variables": ["x"],
        "numerator": _sparse((1, (2,)), (-1, (0,))),
        "denominator": _sparse((1, (1,)), (-1, (0,))),
    }
    scalar = _function(("x",), (1, (1,))).model_dump(mode="json")
    request = RationalLieDerivativeRequest.model_validate(
        {
            "vector_field": {
                "coordinate_axis": ["x"],
                "variance": ["COVARIANT"],
                "components": [nonreduced],
                "retained_nonzero_denominators": [nonreduced["denominator"]],
            },
            "tensor": {
                "coordinate_axis": ["x"],
                "variance": [],
                "components": [scalar],
                "retained_nonzero_denominators": [],
            },
        }
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="rank one and CONTRAVARIANT",
    ) as error:
        lie_derivative(request.vector_field, request.tensor)

    assert error.value.errors()[0]["type"].endswith("vector_signature")


def test_admitted_signature_rejects_a_nonreduced_component_as_noncanonical() -> None:
    nonreduced = {
        "variables": ["x"],
        "numerator": _sparse((1, (2,)), (-1, (0,))),
        "denominator": _sparse((1, (1,)), (-1, (0,))),
    }
    vector = RationalCoordinateTensor.model_validate(
        {
            "coordinate_axis": ["x"],
            "variance": ["CONTRAVARIANT"],
            "components": [nonreduced],
            "retained_nonzero_denominators": [nonreduced["denominator"]],
        }
    )
    scalar = _tensor(("x",), (), (_function(("x",), (1, (1,))),))

    with pytest.raises(
        OperationDomainValidationError,
        match="coprime",
    ) as error:
        lie_derivative(vector, scalar)

    assert error.value.errors()[0]["type"].endswith("component_not_canonical")


def test_zero_tensor_at_the_maximum_dense_shape_is_computed_completely() -> None:
    variables = ("x", "y")
    zero = _zero(variables)
    at_boundary = _tensor(
        variables,
        ("COVARIANT",) * MAX_RATIONAL_TENSOR_RANK,
        (zero,) * MAX_RATIONAL_TENSOR_COMPONENTS,
    )
    zero_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (zero, zero),
    )

    result = lie_derivative(zero_vector, at_boundary)
    assert len(result.lie_derivative.components) == MAX_RATIONAL_TENSOR_COMPONENTS
    assert all(not value.numerator.terms for value in result.lie_derivative.components)


def _maximum_dense_formula_inputs() -> tuple[
    RationalCoordinateTensor, RationalCoordinateTensor
]:
    variables = ("x", "y")
    one = _function(variables, (1, (0, 0)))
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )
    tensor = _tensor(
        variables,
        ("COVARIANT",) * MAX_RATIONAL_TENSOR_RANK,
        (one,) * MAX_RATIONAL_TENSOR_COMPONENTS,
    )
    return vector, tensor


def test_maximum_dense_shape_work_ledger_sums_actual_charge_amounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, tensor = _maximum_dense_formula_inputs()
    charged_units: list[int] = []
    original_charge = lie_bounds._Ledger.charge

    def record_charge(ledger: Any, amount: int) -> None:
        charged_units.append(amount)
        original_charge(ledger, amount)

    monkeypatch.setattr(lie_bounds._Ledger, "charge", record_charge)

    plan = build_lie_derivative_plan(vector, tensor)

    assert charged_units
    assert sum(charged_units) == plan.work_units
    assert plan.work_units <= MAX_LIE_DERIVATIVE_WORK_UNITS


def test_maximum_dense_shape_executes_every_backend_formula_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, tensor = _maximum_dense_formula_inputs()
    plan = build_lie_derivative_plan(vector, tensor)
    executed = {"derivative": 0, "product": 0, "sum": 0, "output_component": 0}

    def count_calls(name: str, function: Any) -> Any:
        def counted(*args: Any, **kwargs: Any) -> Any:
            executed[name] += 1
            return function(*args, **kwargs)

        return counted

    monkeypatch.setattr(
        lie_backend,
        "_differentiate",
        count_calls("derivative", lie_backend._differentiate),
    )
    monkeypatch.setattr(
        lie_backend,
        "_multiply",
        count_calls("product", lie_backend._multiply),
    )
    monkeypatch.setattr(
        lie_backend,
        "_add",
        count_calls("sum", lie_backend._add),
    )
    result = lie_backend.compute_lie_derivative_components(vector, tensor, plan)
    executed["output_component"] = len(result)
    formula_terms = sum(len(component.terms) for component in plan.components)
    dimension = len(vector.coordinate_axis)
    assert_charged_work_parity(
        charged={
            "derivative": dimension * (dimension + len(tensor.components)),
            "product": formula_terms,
            "sum": formula_terms,
            "output_component": len(plan.components),
        },
        executed=executed,
    )
    assert all(rational_function_to_sympy(component) == 8 for component in result)


def test_intermediate_support_is_rejected_before_polynomial_expansion() -> None:
    variables = ("x", "y")
    sparse_grid = tuple(
        (1, exponents)
        for exponents in sorted(
            product(range(0, 64, 4), repeat=2),
            reverse=True,
        )
    )
    value = _function(variables, *sparse_grid)
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (value, _zero(variables)),
    )
    scalar = _tensor(variables, (), (value,))

    with pytest.raises(
        OperationDomainValidationError,
        match="4096-term intermediate budget",
    ) as error:
        lie_derivative(vector, scalar)
    assert error.value.errors()[0]["type"].endswith("intermediate_support")


def test_result_exponent_admission_has_an_accepted_and_rejected_edge() -> None:
    variables = ("x",)

    def rational_power(power: int) -> RationalFunction:
        return _function(
            variables,
            (1, (0,)),
            denominator=((1, (power,)),),
        )

    def power_guard(power: int) -> dict[str, Any]:
        return _guard((1, (power,)))

    accepted_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (rational_power(31),),
        guards=(power_guard(31),),
    )
    accepted_scalar = _tensor(
        variables,
        (),
        (rational_power(32),),
        guards=(power_guard(32),),
    )
    accepted = lie_derivative(accepted_vector, accepted_scalar)
    assert accepted.lie_derivative.components[0].denominator.terms[0].exponents == (64,)

    rejected_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (rational_power(64),),
        guards=(power_guard(64),),
    )
    rejected_scalar = _tensor(
        variables,
        (),
        (rational_power(64),),
        guards=(power_guard(64),),
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="exponent bound 64",
    ) as error:
        lie_derivative(rejected_vector, rejected_scalar)
    assert error.value.errors()[0]["type"].endswith("result_exponent")


def test_profile_round_trip_and_catalog_execution_use_the_same_contract() -> None:
    variables = ("x",)
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (_function(variables, (1, (1,))),),
    )
    scalar = _tensor(variables, (), (_function(variables, (1, (2,))),))
    request = RationalLieDerivativeRequest(vector_field=vector, tensor=scalar)
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "differential_geometry.rational_tensor.lie_derivative.compute"
    )

    result = tool.run(request)
    parsed = RationalLieDerivativeProfile.model_validate(result.model_dump(mode="json"))

    assert parsed == result
    assert _expressions(result.lie_derivative) == (2 * sympy.Symbol("x") ** 2,)


def test_profile_rejects_a_forged_result_that_drops_an_inherited_guard() -> None:
    variables = ("x",)
    x_guard = _guard((1, (1,)))
    rational_vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(
                variables,
                (1, (0,)),
                denominator=((1, (1,)),),
            ),
        ),
        guards=(x_guard,),
    )
    scalar = _tensor(variables, (), (_function(variables, (1, (1,))),))
    forged_result = _tensor(variables, (), (_function(variables, (1, (0,))),))

    with pytest.raises(ValidationError, match="retain exactly"):
        RationalLieDerivativeProfile(
            vector_field=rational_vector,
            source=scalar,
            lie_derivative=forged_result,
        )
