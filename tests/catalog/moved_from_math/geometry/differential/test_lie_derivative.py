"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/differential/test_lie_derivative.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import gcd, lcm, prod
from typing import Any

import sympy

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.differential import (
    RationalCoordinateTensor,
    RationalLieDerivativeProfile,
    verify_lie_derivative,
)
from jacobian.math.geometry.differential._models import RationalLieDerivativeRequest
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COMPONENTS,
    MAX_RATIONAL_TENSOR_RANK,
)
from jacobian.math.polynomials._conversions import (
    rational_function_to_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions import _bounds as rational_bounds
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial


def _sparse(*terms: PolynomialTerm) -> dict[str, Any]:
    return {
        "terms": [
            {
                "coefficient": {
                    "num": (
                        coefficient if isinstance(coefficient, int) else coefficient[0]
                    ),
                    "den": (1 if isinstance(coefficient, int) else coefficient[1]),
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
            "components": [component.model_dump() for component in components],
            "retained_nonzero_denominators": list(guards),
        }
    )


def _expressions(tensor: RationalCoordinateTensor) -> tuple[Any, ...]:
    return tuple(rational_function_to_sympy(value) for value in tensor.components)


def _assert_expression(actual: RationalFunction, expected: Any) -> None:
    assert sympy.cancel(rational_function_to_sympy(actual) - expected) == 0


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
        self.original_fraction_bound = rational_bounds._fraction_bound
        self.original_polynomial_bound = rational_bounds._polynomial_bound
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
    parsed = RationalLieDerivativeProfile.model_validate_json(result.model_dump_json())

    assert parsed == result
    assert verify_lie_derivative(parsed) is True
    forged = parsed.model_dump(mode="json")
    forged["lie_derivative"]["components"][0]["numerator"]["terms"][0][
        "coefficient"
    ] = {"num": "3", "den": "1"}
    forged_claim = RationalLieDerivativeProfile.model_validate_json(json.dumps(forged))
    assert verify_lie_derivative(forged_claim) is False
    assert _expressions(result.lie_derivative) == (2 * sympy.Symbol("x") ** 2,)


type Coefficient = int | tuple[int, int]
type PolynomialTerm = tuple[Coefficient, tuple[int, ...]]
