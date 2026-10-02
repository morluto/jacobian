"""Empty-axis composition bounds, canonical results, and exact cancellations."""

from fractions import Fraction

import pytest
import sympy
from sympy import symbols

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.composition import (
    RationalFunctionMapComposition,
    compose_maps,
)
from jacobian.math.polynomials.rational_functions.composition import (
    operations as composition_ops,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction


def _rf(expression: object, variables: tuple[object, ...]) -> RationalFunction:
    return rational_function_from_sympy(expression, tuple(str(v) for v in variables))


def _map(
    source: tuple[str, ...],
    target: tuple[str, ...],
    components: tuple[RationalFunction, ...],
) -> RationalFunctionMap:
    return RationalFunctionMap(
        source_variables=source, target_coordinates=target, components=components
    )


@pytest.mark.parametrize("exponent", [2, 33, 34, 64])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_constant_composition_height_refusal_is_host_digit_independent(
    exponent: int,
    reciprocal: bool,
) -> None:
    u = symbols("u")
    value = sympy.Rational(1, 10**127) if reciprocal else 10**127
    inner = _map((), ("u",), (_rf(value, ()),))
    outer = _map(("u",), ("v",), (_rf(u**exponent, (u,)),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_maps(outer, inner)
    assert (
        error.value.errors()[0]["type"] == "rational_function_map.compose.result_height"
    )
    assert error.value.errors()[0]["loc"] == ("outer", "inner")


@pytest.mark.parametrize("constant", [0, 10**127, sympy.Rational(1, 10**127)])
def test_constant_composition_accepts_canonical_height_boundary(
    constant: int | sympy.Rational,
) -> None:
    u = symbols("u")
    inner = _map((), ("u",), (_rf(constant, ()),))
    outer = _map(("u",), ("v",), (_rf(u, (u,)),))
    result = compose_maps(outer, inner)
    restored = RationalFunctionMapComposition.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert restored.composite.components == (_rf(constant, ()),)
    assert restored.composite.source_variables == ()
    consumer = _map(("v",), ("w",), (_rf(symbols("v"), (symbols("v"),)),))
    assert compose_maps(consumer, restored.composite).composite.components == (
        _rf(constant, ()),
    )


@pytest.mark.parametrize("divide", [False, True])
def test_constant_composition_preserves_large_intermediate_cancellation(
    divide: bool,
) -> None:
    u, v = symbols("u v")
    inner = _map((), ("u", "v"), (_rf(10**127, ()), _rf(10**127, ())))
    expression = u**34 / v**34 if divide else u**34 - v**34 + 1
    outer = _map(("u", "v"), ("w",), (_rf(expression, (u, v)),))
    assert compose_maps(outer, inner).composite.components == (_rf(1, ()),)


def test_constant_composition_admits_work_before_power_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = symbols("u")
    inner = _map((), ("u",), (_rf(10**127, ()),))
    # One-variable source recognition is cheap, but the scalar powers would
    # exceed the aggregate width/work envelope before exact evaluation.
    outer = _map(("u",), ("v",), (_rf(sum(u**degree for degree in range(65)), (u,)),))

    def unexpected_expansion(*args: object) -> Fraction:
        pytest.fail("constant substitution expanded before aggregate work admission")

    monkeypatch.setattr(
        composition_ops, "_constant_substitute_polynomial", unexpected_expansion
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_maps(outer, inner)
    assert (
        error.value.errors()[0]["type"] == "rational_function_map.compose.work_budget"
    )


def test_constant_addition_bound_covers_fraction_cross_products() -> None:
    u, v = symbols("u v")
    source = _rf(u + v, (u, v))
    values = (Fraction(1, 2**100), Fraction(1, 3**100))
    bound = composition_ops._constant_substitution_bits(
        source.numerator, values, composition_ops._Ledger()
    )
    exact = sum(values, Fraction(0))
    assert bound >= exact.numerator.bit_length() + exact.denominator.bit_length()
    inner = _map(
        (),
        ("u", "v"),
        tuple(
            _rf(sympy.Rational(value.numerator, value.denominator), ())
            for value in values
        ),
    )
    result = compose_maps(_map(("u", "v"), ("w",), (source,)), inner)
    assert (
        result.composite.components[0].numerator.terms[0].coefficient.as_fraction()
        == exact
    )
