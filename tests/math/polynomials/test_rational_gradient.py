"""Quotient identities checked by independent sparse coefficient arithmetic."""

from fractions import Fraction
from random import Random
from time import monotonic

import pytest
from sympy import symbols

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.gradient import (
    RationalFunctionGradient,
    gradient,
)
from jacobian.math.polynomials.rational_functions.gradient import (
    operations as gradient_ops,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_canonical_rational_function,
)

type Coefficients = dict[tuple[int, ...], Fraction]


def _coefficients(value: SparseRationalPolynomial) -> Coefficients:
    return {t.exponents: t.coefficient.as_fraction() for t in value.terms}


def _product(a: Coefficients, b: Coefficients) -> Coefficients:
    out: Coefficients = {}
    for e, c in a.items():
        for f, d in b.items():
            key = tuple(x + y for x, y in zip(e, f, strict=True))
            out[key] = out.get(key, Fraction(0)) + c * d
    return {e: c for e, c in out.items() if c}


def _difference(a: Coefficients, b: Coefficients) -> Coefficients:
    return {
        e: c
        for e in a.keys() | b.keys()
        if (c := a.get(e, Fraction(0)) - b.get(e, Fraction(0)))
    }


def _derivative(a: Coefficients, axis: int) -> Coefficients:
    return {
        tuple(x - int(i == axis) for i, x in enumerate(e)): c * e[axis]
        for e, c in a.items()
        if e[axis]
    }


def _identity(source: RationalFunction) -> RationalFunctionGradient:
    result = gradient(source)
    assert result.source == source
    assert result.variables == source.variables
    p, q = _coefficients(source.numerator), _coefficients(source.denominator)
    for axis, component in enumerate(result.partial_derivatives):
        quotient = _difference(
            _product(_derivative(p, axis), q), _product(p, _derivative(q, axis))
        )
        assert _product(_coefficients(component.numerator), _product(q, q)) == _product(
            quotient, _coefficients(component.denominator)
        )
        assert component.variables == source.variables
        assert require_canonical_rational_function(component) == component
    assert (
        RationalFunctionGradient.model_validate_json(result.model_dump_json()) == result
    )
    return result


def _monomial_source(
    variables: tuple[str, ...],
    numerator: tuple[int, ...],
    denominator: tuple[int, ...],
    coefficient: int = 1,
) -> RationalFunction:
    return RationalFunction(
        variables=variables,
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=numerator,
                ),
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=denominator,
                ),
            )
        ),
    )


def test_multivariate_quotient_and_mixed_partial_composition() -> None:
    x, y = symbols("x y")
    source = rational_function_from_sympy((x * x + y) / (x - y), ("x", "y"))
    result = _identity(source)
    a, b = (
        RationalFunction.model_validate_json(v.model_dump_json())
        for v in result.partial_derivatives
    )
    assert gradient(a).partial_derivatives[1] == gradient(b).partial_derivatives[0]


@pytest.mark.parametrize("weights", [(1, 1, 1), (2, 3, 5), (1, -2, 3)])
def test_low_total_degree_reciprocal_quadratic_power(
    weights: tuple[int, int, int],
) -> None:
    x, y, z = symbols("x y z")
    axes = (x, y, z)
    quadratic = 1 + sum(c * v * v for c, v in zip(weights, axes, strict=True))
    source = rational_function_from_sympy(1 / quadratic**2, ("x", "y", "z"))
    result = _identity(source)
    for c, v, partial in zip(weights, axes, result.partial_derivatives, strict=True):
        expected = rational_function_from_sympy(
            -4 * c * v / quadratic**3, source.variables
        )
        assert partial == expected
        assert len(partial.denominator.terms) == 20


@pytest.mark.parametrize("axis", [(), ("x",), ("x", "y")])
@pytest.mark.parametrize("constant", [0, 3, 10**127])
def test_constants_and_empty_axes(axis: tuple[str, ...], constant: int) -> None:
    source = rational_function_from_sympy(constant, axis)
    result = _identity(source)
    assert len(result.partial_derivatives) == len(axis)
    assert all(not c.numerator.terms for c in result.partial_derivatives)


def test_seeded_small_rational_corpus_by_exact_quotient_identity() -> None:
    x, y = symbols("x y")
    rng = Random(2878)
    for _ in range(20):
        a, b, c, d = [rng.randrange(-3, 4) for _ in range(4)]
        source = rational_function_from_sympy(
            (a * x * x + b * y + c) / (x + y + d), ("x", "y")
        )
        _identity(source)


def test_cancellation_and_axis_permutation() -> None:
    x, y = symbols("x y")
    source = rational_function_from_sympy(1 / (x + y) ** 2, ("x", "y"))
    result = _identity(source)
    expected = rational_function_from_sympy(-2 / (x + y) ** 3, ("x", "y"))
    assert result.partial_derivatives == (expected, expected)
    swapped = _identity(rational_function_from_sympy(1 / (x + y) ** 2, ("y", "x")))
    assert swapped.variables == ("y", "x")


def test_polynomial_sparse_eight_axis_boundary_avoids_dense_expansion() -> None:
    axes = tuple(f"x{i}" for i in range(8))
    source = _monomial_source(axes, (64,) * 8, (0,) * 8)
    result = _identity(source)
    for axis, component in enumerate(result.partial_derivatives):
        assert component.numerator.terms[0].coefficient.num == 64
        assert component.numerator.terms[0].exponents == tuple(
            63 if j == axis else 64 for j in range(8)
        )


def test_monomial_denominator_near_exponent_boundary() -> None:
    result = _identity(_monomial_source(("x", "y"), (0, 64), (63, 0)))
    assert result.partial_derivatives[0].denominator.terms[0].exponents == (64, 0)
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        gradient(_monomial_source(("x", "y"), (0, 64), (64, 0)))


def test_true_output_coefficient_boundary() -> None:
    _identity(_monomial_source(("x",), (64,), (0,), 10**125))
    with pytest.raises(OperationResourceAdmissionError, match="coefficient"):
        gradient(_monomial_source(("x",), (64,), (0,), 10**127))


def test_repeated_linear_denominator_cancels_before_result_exponent() -> None:
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x + 1) ** 33, ("x",))
    result = _identity(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -33 / (x + 1) ** 34, ("x",)
    )
    source = rational_function_from_sympy(1 / (x + 1) ** 35, ("x",))
    result = _identity(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -35 / (x + 1) ** 36, ("x",)
    )
    source = rational_function_from_sympy(1 / ((x + 1) ** 20 * (x + 2) ** 20), ("x",))
    result = _identity(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -20 * (2 * x + 3) / ((x + 1) ** 21 * (x + 2) ** 21), ("x",)
    )
    source = rational_function_from_sympy(1 / ((x + 1) ** 33 * (x + 2)), ("x",))
    result = _identity(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -(34 * x + 67) / ((x + 1) ** 34 * (x + 2) ** 2), ("x",)
    )
    x, y = symbols("x y")
    source = rational_function_from_sympy(1 / (x + y) ** 33, ("x", "y"))
    result = _identity(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -33 / (x + y) ** 34, ("x", "y")
    )
    source = _monomial_source(("x", "y"), (1, 1), (1, 0))
    with pytest.raises(OperationDomainValidationError, match="coprime"):
        gradient(source)
    x, y = symbols("x y")
    p = rational_function_from_sympy((x - y) * (x + 1), ("x", "y"))
    q = rational_function_from_sympy(x - y, ("x", "y"))
    authored = RationalFunction(
        variables=("x", "y"), numerator=p.numerator, denominator=q.numerator
    )
    with pytest.raises(OperationDomainValidationError, match="coprime"):
        gradient(authored)
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        gradient(rational_function_from_sympy(1 / (x + 1) ** 64, ("x",)))


def test_dense_source_box_rejects_before_coprimality_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("coprimality worker must not start")

    monkeypatch.setattr(
        "jacobian.math.polynomials.rational_functions.gradient.operations.recognize_canonical_rational_functions",
        boom,
    )
    axes = tuple(f"x{i}" for i in range(8))
    high = (64,) * 8
    zero = (0,) * 8
    source = RationalFunction(
        variables=axes,
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=zero,
                ),
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=high,
                ),
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=zero,
                ),
            )
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work") as error:
        gradient(source)
    assert error.value.errors()[0]["type"].endswith("work_budget")


def test_univariate_binomial_power_cancellation() -> None:
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x**2 + 1) ** 17, ("x",))
    result = gradient(source)
    expected = rational_function_from_sympy(-34 * x / (x**2 + 1) ** 18, ("x",))
    assert result.partial_derivatives == (expected,)
    x, y = symbols("x y")
    numerator = sum(x ** (2 * i) for i in range(32)) * sum(y**j for j in range(5))
    source = rational_function_from_sympy(numerator / (x + 1) ** 33, ("x", "y"))
    with pytest.raises(OperationResourceAdmissionError, match="term"):
        gradient(source)
    even_grid = sum(x ** (2 * a) * y ** (2 * b) for a in range(16) for b in range(16))
    source = rational_function_from_sympy(even_grid / (x + y) ** 33, ("x", "y"))
    with pytest.raises(OperationResourceAdmissionError, match="term"):
        gradient(source)


def test_shared_request_deadline() -> None:
    source = _monomial_source(("x",), (1,), (0,))
    with (
        request_execution(monotonic() - 100),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        gradient(source)


def test_polar_metric_component_gradient() -> None:
    r, _theta = symbols("r theta")
    result = _identity(rational_function_from_sympy(r * r, ("r", "theta")))
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        2 * r, ("r", "theta")
    )
    assert not result.partial_derivatives[1].numerator.terms


def test_general_gradient_recognizes_and_cancels_in_the_bounded_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_server_gcd(value: RationalFunction) -> RationalFunction:
        raise AssertionError("general-gradient gcd must not run in the server process")

    monkeypatch.setattr(
        gradient_ops, "require_canonical_rational_function", fail_server_gcd
    )
    x, y = symbols("x y")
    _identity(rational_function_from_sympy((x * x + y) / (x - y), ("x", "y")))


def test_inactive_axis_of_a_nonmonomial_reciprocal_is_canonical_zero() -> None:
    x, _y = symbols("x y")
    result = _identity(rational_function_from_sympy(1 / (x + 1), ("x", "y")))
    assert not result.partial_derivatives[1].numerator.terms
    assert result.partial_derivatives[1].denominator.terms[0].coefficient.num == 1
    assert result.partial_derivatives[1].denominator.terms[0].exponents == (0, 0)


def test_inactive_axes_of_a_bivariate_power_skip_denominator_gcds() -> None:
    axis = tuple(f"x{index}" for index in range(8))
    names = symbols("x0:8")
    linear = names[0] + names[1] + 1
    result = _identity(rational_function_from_sympy(1 / linear**20, axis))
    assert all(
        not component.numerator.terms for component in result.partial_derivatives[2:]
    )
    assert all(
        component.numerator.terms for component in result.partial_derivatives[:2]
    )
