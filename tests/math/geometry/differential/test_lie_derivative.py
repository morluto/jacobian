"""Exact contract tests for rational coordinate-tensor Lie derivatives."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd, lcm, prod
from time import monotonic
from typing import Any

import pytest
import sympy
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._execution import (
    OperationExecutionTimeoutError,
    current_request_execution,
    request_execution,
)
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
from jacobian.math.geometry.differential import operations as lie_operations
from jacobian.math.geometry.differential._bounds import (
    MAX_LIE_DERIVATIVE_WORK_UNITS,
    build_lie_derivative_plan,
)
from jacobian.math.geometry.differential._execution import (
    LIE_DERIVATIVE_WALL_SECONDS,
    require_lie_derivative_deadline,
)
from jacobian.math.geometry.differential._recognition_process import (
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    MAX_RATIONAL_TENSOR_RANK,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial

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


def test_maximum_dense_shape_skips_trivial_coprimality_gcds() -> None:
    vector, tensor = _maximum_dense_formula_inputs()
    plan = build_lie_derivative_plan(vector, tensor)

    assert not plan.recognition_candidates
    assert dict(plan.work_units_by_category)["recognition"] == 0
    assert sum(dict(plan.work_units_by_category).values()) == plan.work_units
    assert plan.work_units <= MAX_LIE_DERIVATIVE_WORK_UNITS


def _categorized_accounting_inputs() -> tuple[
    RationalCoordinateTensor, RationalCoordinateTensor
]:
    variables = ("x", "y")
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(
                variables,
                (1, (1, 0)),
                denominator=((1, (0, 1)), (1, (0, 0))),
            ),
            _function(
                variables,
                (1, (0, 1)),
                denominator=((1, (1, 0)), (1, (0, 0))),
            ),
        ),
        guards=(
            _guard((1, (0, 1)), (1, (0, 0))),
            _guard((1, (1, 0)), (1, (0, 0))),
        ),
    )
    covector = _tensor(
        variables,
        ("COVARIANT",),
        (
            _function(variables, (1, (1, 0)), (1, (0, 1))),
            _function(variables, (1, (1, 0)), (-1, (0, 1))),
        ),
    )
    return vector, covector


def _dense_source_coefficients(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> int:
    if not polynomial.terms:
        return 1
    return prod(
        max(term.exponents[axis] for term in polynomial.terms) + 1
        for axis in range(variable_count)
    )


def _observed_recognition_units(value: RationalFunction) -> int:
    variable_count = len(value.variables)
    numerator_dense = _dense_source_coefficients(value.numerator, variable_count)
    denominator_dense = _dense_source_coefficients(value.denominator, variable_count)
    degree_steps = (
        sum(
            max(
                max(term.exponents[axis] for term in value.numerator.terms),
                max(term.exponents[axis] for term in value.denominator.terms),
            )
            for axis in range(variable_count)
        )
        + 1
    )
    coefficient_digits = max(
        len(str(abs(term.coefficient.as_integer_ratio()[0])))
        + len(str(term.coefficient.as_integer_ratio()[1]))
        for polynomial in (value.numerator, value.denominator)
        for term in polynomial.terms
    )
    coefficient_chunks = max(1, (coefficient_digits + 31) // 32)
    return (numerator_dense + denominator_dense) * degree_steps * coefficient_chunks


def _observed_normalization_units(expression: Any, variables: tuple[str, ...]) -> int:
    symbols = tuple(sympy.Symbol(variable) for variable in variables)
    numerator_expression, denominator_expression = sympy.fraction(expression)
    numerator = sympy.Poly(numerator_expression, *symbols, domain=sympy.QQ)
    denominator = sympy.Poly(denominator_expression, *symbols, domain=sympy.QQ)
    if numerator.is_zero:
        return 1

    def dense_coefficients(polynomial: Any) -> int:
        return prod(int(degree) + 1 for degree in polynomial.degree_list())

    coefficient_digits = max(
        len(str(abs(int(coefficient.p)))) + len(str(int(coefficient.q)))
        for polynomial in (numerator, denominator)
        for coefficient in polynomial.coeffs()
    )
    dense_total = dense_coefficients(numerator) + dense_coefficients(denominator)
    return dense_total * (dense_total - 1) * coefficient_digits


class _SourceConversionObserver:
    def __init__(self, executed: dict[str, int], calls: dict[str, int]) -> None:
        self.executed = executed
        self.calls = calls
        self.observing_bound = False
        self.original_fraction_bound = lie_bounds._fraction_bound
        self.original_polynomial_bound = lie_bounds._polynomial_bound
        self.original_fraction = Fraction
        self.original_gcd = gcd
        self.original_lcm = lcm
        self.original_sparse_conversion = sparse_rational_polynomial_to_sympy

    def bound(self, value: RationalFunction, ledger: Any) -> Any:
        self.calls["source_conversion"] += 1
        self.observing_bound = True
        try:
            return self.original_fraction_bound(value, ledger)
        finally:
            self.observing_bound = False

    def polynomial_bound(self, polynomial: SparseRationalPolynomial) -> Any:
        result = self.original_polynomial_bound(polynomial)
        if self.observing_bound:
            terms = len(polynomial.terms)
            variables = len(polynomial.terms[0].exponents)
            # Integral scaling, primitive division, coefficient-height
            # inspection, and maximum/minimum exponent scans.
            self.executed["source_conversion"] += terms * (3 + 2 * variables)
        return result

    def fraction(self, *args: Any, **kwargs: Any) -> Any:
        if self.observing_bound:
            self.executed["source_conversion"] += 1
        return self.original_fraction(*args, **kwargs)

    def gcd(self, *integers: int) -> int:
        if self.observing_bound:
            self.executed["source_conversion"] += len(integers)
        return self.original_gcd(*integers)

    def lcm(self, *integers: int) -> int:
        if self.observing_bound:
            self.executed["source_conversion"] += len(integers)
        return self.original_lcm(*integers)

    def convert(
        self, polynomial: SparseRationalPolynomial, variables: tuple[str, ...]
    ) -> Any:
        self.calls["source_conversion"] += 1
        result = self.original_sparse_conversion(polynomial, variables)
        coefficient_chunks = sum(
            max(
                1,
                (len(str(abs(int(coefficient.p)))) + len(str(int(coefficient.q))) + 31)
                // 32,
            )
            for coefficient in result.coeffs()
        )
        dense_coefficients = (
            1
            if result.is_zero
            else prod(int(degree) + 1 for degree in result.degree_list())
        )
        self.executed["source_conversion"] += coefficient_chunks + dense_coefficients
        return result


class _PolynomialArithmeticObserver:
    def __init__(self, executed: dict[str, int], calls: dict[str, int]) -> None:
        self.executed = executed
        self.calls = calls
        self.active = 0
        self.polynomial_type = type(sympy.Poly(sympy.Symbol("x"), sympy.Symbol("x")))
        self.original_multiply = self.polynomial_type.__mul__
        self.original_add = self.polynomial_type.__add__
        self.original_subtract = self.polynomial_type.__sub__

    def run(self, function: Any, *args: Any) -> Any:
        self.active += 1
        try:
            return function(*args)
        finally:
            self.active -= 1

    def multiply(self, left: Any, right: Any) -> Any:
        if (
            self.active
            and isinstance(right, self.polynomial_type)
            and not left.is_zero
            and not right.is_zero
        ):
            self.calls["multiplication"] += 1
            self.executed["multiplication"] += len(left.terms()) * len(right.terms())
        return self.original_multiply(left, right)

    def add(self, left: Any, right: Any) -> Any:
        self._observe_addition(left, right)
        return self.original_add(left, right)

    def subtract(self, left: Any, right: Any) -> Any:
        self._observe_addition(left, right)
        return self.original_subtract(left, right)

    def _observe_addition(self, left: Any, right: Any) -> None:
        if (
            self.active
            and isinstance(right, self.polynomial_type)
            and not left.is_zero
            and not right.is_zero
        ):
            self.calls["addition"] += 1
            self.executed["addition"] += len(left.terms()) + len(right.terms())


def test_work_categories_cover_observed_backend_primitives_and_detect_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, tensor = _categorized_accounting_inputs()
    captured_plan: list[Any] = []
    executed = {
        "recognition": 0,
        "source_conversion": 0,
        "differentiation": 0,
        "multiplication": 0,
        "addition": 0,
        "normalization": 0,
    }
    top_level_calls = dict.fromkeys(executed, 0)
    source_observer = _SourceConversionObserver(executed, top_level_calls)
    arithmetic_observer = _PolynomialArithmeticObserver(executed, top_level_calls)

    original_plan = build_lie_derivative_plan
    original_recognition = recognize_canonical_rational_functions
    original_differentiate = lie_backend._differentiate
    original_multiply = lie_backend._multiply
    original_add = lie_backend._add
    original_normalize = rational_function_from_sympy

    def planning(*args: Any, **kwargs: Any) -> Any:
        plan = original_plan(*args, **kwargs)
        captured_plan.append(plan)
        return plan

    def recognizing(*args: Any, **kwargs: Any) -> Any:
        result = original_recognition(*args, **kwargs)
        candidates = args[0]
        top_level_calls["recognition"] += result.recognized_candidates
        executed["recognition"] += sum(
            _observed_recognition_units(candidate.value)
            for candidate in candidates[: result.recognized_candidates]
        )
        return result

    def differentiating(source: Any, axis: int) -> Any:
        top_level_calls["differentiation"] += 1
        executed["differentiation"] += len(source.numerator.terms()) + len(
            source.denominator.terms()
        )
        return arithmetic_observer.run(original_differentiate, source, axis)

    def multiplying(left: Any, right: Any) -> Any:
        return arithmetic_observer.run(original_multiply, left, right)

    def adding(left: Any, right: Any) -> Any:
        return arithmetic_observer.run(original_add, left, right)

    def normalizing(expression: Any, variables: tuple[str, ...], **kwargs: Any) -> Any:
        top_level_calls["normalization"] += 1
        executed["normalization"] += _observed_normalization_units(
            expression, variables
        )
        return original_normalize(expression, variables, **kwargs)

    monkeypatch.setattr(lie_operations, "build_lie_derivative_plan", planning)
    monkeypatch.setattr(
        lie_bounds, "recognize_canonical_rational_functions", recognizing
    )
    monkeypatch.setattr(lie_bounds, "_fraction_bound", source_observer.bound)
    monkeypatch.setattr(
        lie_bounds, "_polynomial_bound", source_observer.polynomial_bound
    )
    monkeypatch.setattr(lie_bounds, "Fraction", source_observer.fraction)
    monkeypatch.setattr(lie_bounds, "gcd", source_observer.gcd)
    monkeypatch.setattr(lie_bounds, "lcm", source_observer.lcm)
    monkeypatch.setattr(
        lie_backend,
        "sparse_rational_polynomial_to_sympy",
        source_observer.convert,
    )
    monkeypatch.setattr(lie_backend, "_differentiate", differentiating)
    monkeypatch.setattr(lie_backend, "_multiply", multiplying)
    monkeypatch.setattr(lie_backend, "_add", adding)
    monkeypatch.setattr(
        arithmetic_observer.polynomial_type,
        "__mul__",
        lambda left, right: arithmetic_observer.multiply(left, right),
    )
    monkeypatch.setattr(
        arithmetic_observer.polynomial_type,
        "__add__",
        lambda left, right: arithmetic_observer.add(left, right),
    )
    monkeypatch.setattr(
        arithmetic_observer.polynomial_type,
        "__sub__",
        lambda left, right: arithmetic_observer.subtract(left, right),
    )
    monkeypatch.setattr(lie_backend, "rational_function_from_sympy", normalizing)

    result = lie_derivative(vector, tensor)

    assert len(captured_plan) == 1
    plan = captured_plan[0]
    charged = dict(plan.work_units_by_category)
    assert all(amount > 0 for amount in executed.values())
    assert all(executed[category] > top_level_calls[category] for category in executed)
    assert sum(charged.values()) == plan.work_units
    assert_charged_work_parity(charged=charged, executed=executed)

    for category, calls in top_level_calls.items():
        collapsed = dict(charged)
        collapsed[category] = calls
        with pytest.raises(AssertionError, match="exceeds its admission charge"):
            assert_charged_work_parity(charged=collapsed, executed=executed)
    omitted = dict(charged)
    del omitted["addition"]
    with pytest.raises(AssertionError, match="has no admission charge"):
        assert_charged_work_parity(charged=omitted, executed=executed)

    assert len(result.lie_derivative.components) == 2


def test_source_conversion_is_precharged_before_content_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("x",)
    one = _function(variables, (1, (0,)))
    vector = _tensor(variables, ("CONTRAVARIANT",), (one,))
    scalar = _tensor(variables, (), (one,))

    def forbidden_content_arithmetic(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unadmitted source reached coefficient arithmetic")

    monkeypatch.setattr(lie_bounds, "MAX_LIE_DERIVATIVE_WORK_UNITS", 1)
    monkeypatch.setattr(lie_bounds, "_polynomial_bound", forbidden_content_arithmetic)

    with pytest.raises(OperationDomainValidationError) as error:
        build_lie_derivative_plan(vector, scalar)

    assert error.value.errors()[0]["type"].endswith("work_budget")


def test_intermediate_support_is_rejected_before_coprimality_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def forbidden_recognition(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("doomed intermediate request reached recognition")

    monkeypatch.setattr(
        lie_bounds,
        "recognize_canonical_rational_functions",
        forbidden_recognition,
    )

    with pytest.raises(
        OperationDomainValidationError,
        match="4096-term intermediate budget",
    ) as error:
        lie_derivative(vector, scalar)
    assert error.value.errors()[0]["type"].endswith("intermediate_support")


def test_work_budget_rejection_precedes_coprimality_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("x", "y")
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (
            _function(variables, (1, (1, 0))),
            _function(variables, (1, (0, 1))),
        ),
    )
    value = _function(variables, (1, (8, 8)))
    tensor = _tensor(
        variables,
        ("COVARIANT",) * MAX_RATIONAL_TENSOR_RANK,
        (value,) * MAX_RATIONAL_TENSOR_COMPONENTS,
    )

    def forbidden_recognition(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("doomed work request reached recognition")

    monkeypatch.setattr(
        lie_bounds,
        "recognize_canonical_rational_functions",
        forbidden_recognition,
    )

    with pytest.raises(OperationDomainValidationError, match="work budget") as error:
        lie_derivative(vector, tensor)

    assert error.value.errors()[0]["type"].endswith("work_budget")


def _result_byte_rejection_inputs() -> tuple[
    RationalCoordinateTensor, RationalCoordinateTensor
]:
    variables = ("x", "y")
    one = _function(variables, (1, (0, 0)))
    vector = _tensor(variables, ("CONTRAVARIANT",), (one, one))
    nonconstant_exponents = tuple(
        exponent
        for exponent in sorted(
            product(range(MAX_RATIONAL_TENSOR_EXPONENT + 1), repeat=2),
            reverse=True,
        )
        if exponent != (0, 0)
    )[: MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS // 2 - 1]
    guards = tuple(
        SparseRationalPolynomial.model_validate(
            _sparse(
                *((1, exponent) for exponent in nonconstant_exponents),
                (index + 1, (0, 0)),
            )
        )
        for index in range(MAX_RATIONAL_TENSOR_LOCUS_GUARDS)
    )
    ordered_guards = canonical_locus_guards(guards, variable_count=2)
    scalar = RationalCoordinateTensor(
        coordinate_axis=variables,
        variance=(),
        components=(one,),
        retained_nonzero_denominators=ordered_guards,
    )
    return vector, scalar


def test_result_byte_rejection_precedes_coprimality_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, scalar = _result_byte_rejection_inputs()

    def forbidden_recognition(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("doomed output request reached recognition")

    monkeypatch.setattr(
        lie_bounds,
        "recognize_canonical_rational_functions",
        forbidden_recognition,
    )

    with pytest.raises(
        OperationDomainValidationError, match="canonical output budget"
    ) as error:
        lie_derivative(vector, scalar)

    assert error.value.errors()[0]["type"].endswith("result_bytes")


def test_dispatch_start_owns_one_deadline_through_result_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, tensor = _categorized_accounting_inputs()
    started = monotonic()
    observed: list[tuple[str, float, float | None]] = []
    original_check = require_lie_derivative_deadline

    def checking(deadline: float, stage: str) -> None:
        execution = current_request_execution()
        observed.append(
            (stage, deadline, execution.deadline if execution is not None else None)
        )
        original_check(deadline, stage)

    monkeypatch.setattr(
        lie_operations,
        "require_lie_derivative_deadline",
        checking,
    )

    with request_execution(started):
        result = lie_derivative(vector, tensor)
        bound = current_request_execution()
        assert bound is not None
        assert bound.deadline == started + LIE_DERIVATIVE_WALL_SECONDS

    assert result.lie_derivative.components
    assert observed
    assert {deadline for _, deadline, _ in observed} == {
        started + LIE_DERIVATIVE_WALL_SECONDS
    }
    assert {bound_deadline for _, _, bound_deadline in observed} == {
        started + LIE_DERIVATIVE_WALL_SECONDS
    }
    assert observed[-1][0] == "after result-size serialization"


def test_expired_dispatch_deadline_stops_before_semantic_preflight() -> None:
    variables = ("x",)
    vector = _tensor(
        variables,
        ("CONTRAVARIANT",),
        (_function(variables, (1, (1,))),),
    )
    scalar = _tensor(variables, (), (_function(variables, (1, (1,))),))

    with (
        request_execution(monotonic() - LIE_DERIVATIVE_WALL_SECONDS),
        pytest.raises(OperationExecutionTimeoutError, match="semantic preflight"),
    ):
        lie_derivative(vector, scalar)


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
