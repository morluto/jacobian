"""Exact rational-map composition checked by sparse evaluation oracles."""

from fractions import Fraction
from itertools import product
from time import monotonic

import pytest
from sympy import symbols

from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.composition import (
    RationalFunctionMapComposition,
    compose_maps,
)
from jacobian.math.polynomials.rational_functions.composition._models import (
    RationalMapCompositionRequest,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction, SparseRationalPolynomial


def _rf(expression: object, variables: tuple[object, ...]) -> RationalFunction:
    return rational_function_from_sympy(expression, tuple(str(v) for v in variables))


def _map(
    source: tuple[str, ...],
    target: tuple[str, ...],
    components: tuple[RationalFunction, ...],
) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=source,
        target_coordinates=target,
        components=components,
    )


def _evaluate_polynomial(
    polynomial: SparseRationalPolynomial, point: tuple[Fraction, ...]
) -> Fraction:
    result = Fraction(0)
    for term in polynomial.terms:
        coefficient = term.coefficient.as_fraction()
        for coordinate, exponent in zip(point, term.exponents, strict=True):
            coefficient *= coordinate**exponent
        result += coefficient
    return result


def _evaluate(value: RationalFunction, point: tuple[Fraction, ...]) -> Fraction:
    numerator = _evaluate_polynomial(value.numerator, point)
    denominator = _evaluate_polynomial(value.denominator, point)
    if not denominator:
        raise ZeroDivisionError
    return numerator / denominator


def test_example_composition_retains_all_construction_guards() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf(x / (x - 1), (x,)), _rf((x + 1) / (x - 2), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z1", "z2"),
        (_rf(y1 + y2, (y1, y2)), _rf(y1 / y2, (y1, y2))),
    )

    result = compose_maps(outer, inner)

    assert result.composite.source_variables == ("x",)
    assert result.composite.target_coordinates == ("z1", "z2")
    assert result.composite.components[0] == _rf(
        (2 * x**2 - 2 * x - 1) / (x**2 - 3 * x + 2), (x,)
    )
    assert result.composite.components[1] == _rf((x**2 - 2 * x) / (x**2 - 1), (x,))
    guard_polynomials = tuple(
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in guard.polynomial.terms
        )
        for guard in result.construction_locus_guard
    )
    assert guard_polynomials == (
        (((1,), Fraction(1)), ((0,), Fraction(-1))),
        (((1,), Fraction(1)), ((0,), Fraction(-2))),
        (((1,), Fraction(1)), ((0,), Fraction(1))),
    )
    assert (
        RationalFunctionMapComposition.model_validate_json(result.model_dump_json())
        == result
    )


def test_cancellation_does_not_erase_stricter_locus() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(x / (x - 1), (x,)),))
    outer = _map(("y",), ("z",), (_rf(1 / y, (y,)),))
    result = compose_maps(outer, inner)

    assert result.composite.components[0] == _rf((x - 1) / x, (x,))
    assert {
        tuple(
            (term.exponents, term.coefficient.as_fraction())
            for term in guard.polynomial.terms
        )
        for guard in result.construction_locus_guard
    } == {
        (((1,), Fraction(1)),),
        (((1,), Fraction(1)), ((0,), Fraction(-1))),
    }


def test_outer_denominator_zero_after_substitution_is_domain_error() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(1, (x,)),))
    outer = _map(("y",), ("z",), (_rf(1 / (y - 1), (y,)),))

    with pytest.raises(OperationDomainValidationError, match="vanishes identically"):
        compose_maps(outer, inner)


def test_permuted_scaled_monomial_map_preserves_sparse_structure() -> None:
    x, y, u, v = symbols("x y u v")
    inner = _map(
        ("x", "y"),
        ("u", "v"),
        (_rf(y, (x, y)), _rf(2 * x, (x, y))),
    )
    outer = _map(("u", "v"), ("z",), (_rf(u**64 + v, (u, v)),))

    result = compose_maps(outer, inner)
    expected = _rf(y**64 + 2 * x, (x, y))
    assert result.composite.components == (expected,)


def test_monomial_fast_path_handles_every_outer_component() -> None:
    x, y, u, v = symbols("x y u v")
    inner = _map(
        ("x", "y"),
        ("u", "v"),
        (_rf(y, (x, y)), _rf(2 * x, (x, y))),
    )
    outer = _map(
        ("u", "v"),
        ("z1", "z2"),
        (_rf(u**3 + v, (u, v)), _rf(u - v**2, (u, v))),
    )

    result = compose_maps(outer, inner)

    assert result.composite.components == (
        _rf(y**3 + 2 * x, (x, y)),
        _rf(y - 4 * x**2, (x, y)),
    )


def test_sparse_eight_axis_degree_sixty_four_composition_is_admitted() -> None:
    source = symbols("x0:8")
    intermediate = symbols("y0:8")
    target = symbols("z0:8")
    inner = _map(
        tuple(map(str, source)),
        tuple(map(str, intermediate)),
        tuple(_rf(value, source) for value in source),
    )
    outer = _map(
        tuple(map(str, intermediate)),
        tuple(map(str, target)),
        tuple(_rf(value**64, intermediate) for value in intermediate),
    )

    result = compose_maps(outer, inner)

    assert all(
        component.numerator.terms[0].exponents
        == tuple(64 * (axis == index) for axis in range(8))
        for index, component in enumerate(result.composite.components)
    )


def test_empty_inner_axis_composes_exact_constants() -> None:
    y = symbols("y")
    inner = _map((), ("y",), (_rf(2, ()),))
    outer = _map(("y",), ("z",), (_rf(y**3 - y, (y,)),))

    result = compose_maps(outer, inner)
    assert result.composite.source_variables == ()
    assert result.composite.components[0] == _rf(6, ())
    assert result.construction_locus_guard == ()


def test_composition_matches_independent_fraction_oracle() -> None:
    x = symbols("x")
    y1, y2 = symbols("y1 y2")
    inner = _map(
        ("x",),
        ("y1", "y2"),
        (_rf((x + 1) / (x - 3), (x,)), _rf((x - 2) / (x + 2), (x,))),
    )
    outer = _map(
        ("y1", "y2"),
        ("z1", "z2"),
        (_rf(y1**2 + y2, (y1, y2)), _rf((y1 - y2) / (y1 + y2), (y1, y2))),
    )
    result = compose_maps(outer, inner)

    for numerator, denominator in product(range(-4, 5), repeat=2):
        point = (Fraction(numerator, denominator or 1),)
        try:
            inner_values = tuple(
                _evaluate(component, point) for component in inner.components
            )
            expected = (
                inner_values[0] ** 2 + inner_values[1],
                (inner_values[0] - inner_values[1])
                / (inner_values[0] + inner_values[1]),
            )
            actual = tuple(
                _evaluate(component, point) for component in result.composite.components
            )
        except ZeroDivisionError:
            continue
        assert actual == expected


def test_axis_mismatch_and_authored_noncanonical_source_are_rejected() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("u",), (_rf(x, (x,)),))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with pytest.raises(ValueError, match="must equal"):
        RationalMapCompositionRequest(outer=outer, inner=inner)

    authored = RationalFunction(
        variables=("x",),
        numerator=_rf((x - 1) * (x + 1), (x,)).numerator,
        denominator=_rf(x - 1, (x,)).numerator,
    )
    bad_inner = _map(("x",), ("y",), (authored,))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with pytest.raises(OperationDomainValidationError, match="coprime"):
        compose_maps(outer, bad_inner)


def test_shared_deadline_is_honored() -> None:
    x, y = symbols("x y")
    inner = _map(("x",), ("y",), (_rf(x, (x,)),))
    outer = _map(("y",), ("z",), (_rf(y, (y,)),))
    with (
        request_execution(monotonic() - 100),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        compose_maps(outer, inner)
