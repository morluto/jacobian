"""Domain functions for polynomial vector calculus operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import sympy
from pydantic_core import PydanticCustomError

from jacobian._exact import require_bounded_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.polynomials.vector_calculus._models import (
    _MAX_COEFFICIENT_DIGITS,
    _MAX_TERMS,
    CurlRequest,
    DirectionalDerivativeRequest,
    ScalarFieldRequest,
    ScalarResult,
    VectorFieldRequest,
    VectorResult,
    _require_field_polynomial,
)


def _run_admission(
    admission: Callable[[], None], *, location: tuple[str | int, ...]
) -> None:
    """Translate field admission failures into typed operation diagnostics."""

    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="polynomial_vector_calc.admission",
            message=str(exc),
        ) from exc


def _admit_field_polynomial(
    polynomial: RationalPolynomial,
    *,
    label: str,
    location: tuple[str | int, ...],
) -> None:
    """Apply the field polynomial envelope with a precise source location."""

    _run_admission(
        lambda: _require_field_polynomial(polynomial, label=label),
        location=location,
    )


def _admit_scalar_field(polynomial: RationalPolynomial) -> None:
    _admit_field_polynomial(
        polynomial,
        label="scalar field",
        location=("polynomial",),
    )
    if len(polynomial.polynomial.terms) * len(polynomial.variables) > _MAX_TERMS:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial_vector_calc.derivative_term_budget",
            message="scalar-field derivatives exceed the result-term budget",
        )


def _admit_vector_field(components: tuple[RationalPolynomial, ...]) -> None:
    for index, component in enumerate(components):
        _admit_field_polynomial(
            component,
            label="vector-field component",
            location=("components", index),
        )
    if sum(len(item.polynomial.terms) for item in components) > _MAX_TERMS:
        raise OperationDomainValidationError(
            location=("components",),
            code="polynomial_vector_calc.derivative_term_budget",
            message="vector-field derivatives exceed the result-term budget",
        )


def _wire(expression: sympy.Expr, variables: tuple[str, ...]) -> RationalPolynomial:
    return rational_polynomial_from_sympy(
        sympy.Poly(
            sympy.expand(expression), *symbols_for_variables(variables), domain=sympy.QQ
        ),
        variables,
        maximum_terms=256,
    )


def _expressions(
    polynomials: Iterable[RationalPolynomial],
) -> tuple[sympy.Expr, ...]:
    return tuple(rational_polynomial_to_sympy(item).as_expr() for item in polynomials)


def compute_gradient(request: ScalarFieldRequest) -> VectorResult:
    _admit_scalar_field(request.polynomial)
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    return VectorResult(
        components=tuple(
            _wire(sympy.diff(expression, variable), variables)
            for variable in symbols_for_variables(variables)
        ),
    )


def compute_divergence(request: VectorFieldRequest) -> ScalarResult:
    _admit_vector_field(request.components)
    variables = request.components[0].variables
    expression = sum(
        sympy.diff(component, variable)
        for component, variable in zip(
            _expressions(request.components),
            symbols_for_variables(variables),
            strict=True,
        )
    )
    return ScalarResult(
        result=_wire(expression, variables),
    )


def compute_curl(request: CurlRequest) -> VectorResult:
    """Return the standard three-dimensional curl of a polynomial field."""

    _admit_vector_field(request.components)
    variables = request.components[0].variables
    x, y, z = symbols_for_variables(variables)
    fx, fy, fz = _expressions(request.components)
    return VectorResult(
        components=(
            _wire(sympy.diff(fz, y) - sympy.diff(fy, z), variables),
            _wire(sympy.diff(fx, z) - sympy.diff(fz, x), variables),
            _wire(sympy.diff(fy, x) - sympy.diff(fx, y), variables),
        ),
    )


def compute_laplacian(request: ScalarFieldRequest) -> ScalarResult:
    _admit_scalar_field(request.polynomial)
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    laplacian = sum(
        sympy.diff(expression, variable, 2)
        for variable in symbols_for_variables(variables)
    )
    return ScalarResult(
        result=_wire(laplacian, variables),
    )


def compute_directional_derivative(
    request: DirectionalDerivativeRequest,
) -> ScalarResult:
    _admit_scalar_field(request.polynomial)
    for index, coordinate in enumerate(request.direction):
        try:
            require_bounded_rational(
                coordinate,
                max_digits=_MAX_COEFFICIENT_DIGITS,
                label="direction coordinate",
            )
        except ValueError as exc:
            raise OperationDomainValidationError(
                location=("direction", index),
                code="polynomial_vector_calc.direction_coordinate_bound",
                message=str(exc),
            ) from exc
    variables = request.polynomial.variables
    expression = rational_polynomial_to_sympy(request.polynomial).as_expr()
    gradient = (
        sympy.diff(expression, variable)
        for variable in symbols_for_variables(variables)
    )
    direction = (
        sympy.Rational(*coordinate.as_integer_ratio())
        for coordinate in request.direction
    )
    return ScalarResult(
        result=_wire(
            sum(
                derivative * coordinate
                for derivative, coordinate in zip(gradient, direction, strict=True)
            ),
            variables,
        ),
    )


__all__ = [
    "compute_curl",
    "compute_directional_derivative",
    "compute_divergence",
    "compute_gradient",
    "compute_laplacian",
]
