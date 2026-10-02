"""Signed Lie admission preserves exact brackets, sources, and inherited loci."""

from fractions import Fraction

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential import (
    RationalCoordinateTensor,
    RationalLieDerivativeProfile,
    lie_derivative,
    verify_lie_derivative,
)
from jacobian.math.geometry.differential import _bounds as bounds
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _function(
    axis: tuple[str, ...],
    terms: tuple[tuple[Fraction, tuple[int, ...]], ...],
    denominator: tuple[int, ...] | None = None,
) -> RationalFunction:
    return RationalFunction(
        variables=axis,
        numerator=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(c), exponents=e
                )
                for c, e in terms
            )
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1),
                    exponents=denominator or (0,) * len(axis),
                ),
            )
        ),
    )


def _vector(
    exponent: int,
    coefficient: int = 1,
    *,
    reciprocal: bool = False,
    extra: bool = False,
) -> RationalCoordinateTensor:
    axis = ("x",)
    value = _function(
        axis,
        ((Fraction(coefficient), (0 if reciprocal else exponent,)),)
        + (((Fraction(1), (exponent - 1,)),) if extra else ()),
        (exponent,) if reciprocal else None,
    )
    return RationalCoordinateTensor(
        coordinate_axis=axis,
        variance=("CONTRAVARIANT",),
        components=(value,),
        retained_nonzero_denominators=(value.denominator,) if reciprocal else (),
    )


def _expression(value: RationalFunction) -> sympy.Expr:
    axes = tuple(sympy.Symbol(name) for name in value.variables)

    def polynomial(poly: SparseRationalPolynomial) -> sympy.Expr:
        return sum(
            (
                sympy.Rational(term.coefficient.num, term.coefficient.den)
                * sympy.prod(
                    axis**exponent
                    for axis, exponent in zip(axes, term.exponents, strict=True)
                )
                for term in poly.terms
            ),
            sympy.S.Zero,
        )

    return polynomial(value.numerator) / polynomial(value.denominator)


def _assert_bracket(
    vector: RationalCoordinateTensor, tensor: RationalCoordinateTensor
) -> RationalLieDerivativeProfile:
    result = lie_derivative(vector, tensor)
    restored = RationalLieDerivativeProfile.model_validate_json(
        result.model_dump_json()
    )
    assert verify_lie_derivative(restored)
    assert restored.vector_field == vector
    assert restored.source == tensor
    assert restored.lie_derivative.coordinate_axis == tensor.coordinate_axis
    assert restored.lie_derivative.variance == tensor.variance
    for component, actual in enumerate(restored.lie_derivative.components):
        expected = sum(
            (
                _expression(vector.components[axis])
                * sympy.diff(_expression(tensor.components[component]), name)
                - _expression(tensor.components[axis])
                * sympy.diff(_expression(vector.components[component]), name)
                for axis, name in enumerate(tensor.coordinate_axis)
            ),
            sympy.S.Zero,
        )
        assert sympy.cancel(_expression(actual) - expected) == 0
    return restored


@pytest.mark.parametrize("exponent", [32, 33, 64, 128])
def test_polynomial_self_brackets_are_zero(exponent: int) -> None:
    vector = _vector(exponent)
    result = _assert_bracket(vector, vector)
    assert not result.lie_derivative.components[0].numerator.terms
    assert not result.lie_derivative.retained_nonzero_denominators
    handoff = lie_derivative(_vector(128), result.lie_derivative)
    assert handoff.lie_derivative == result.lie_derivative


@pytest.mark.parametrize("exponent", [22, 32, 63, 64])
def test_reciprocal_proportional_brackets_retain_original_locus(exponent: int) -> None:
    vector, tensor = (
        _vector(exponent, 7, reciprocal=True),
        _vector(exponent, 21, reciprocal=True),
    )
    result = _assert_bracket(vector, tensor)
    assert not result.lie_derivative.components[0].numerator.terms
    assert (
        result.lie_derivative.retained_nonzero_denominators
        == vector.retained_nonzero_denominators
    )
    assert (
        lie_derivative(vector, result.lie_derivative).lie_derivative
        == result.lie_derivative
    )


def test_proportional_and_partially_shared_support_use_signed_coefficients() -> None:
    assert (
        not _assert_bracket(_vector(33, 7), _vector(33, 21))
        .lie_derivative.components[0]
        .numerator.terms
    )
    partial = _assert_bracket(_vector(33), _vector(33, extra=True))
    assert partial.lie_derivative.components[0] == _function(
        ("x",), ((Fraction(-1), (64,)),)
    )


@pytest.mark.parametrize("axis", [("x", "y"), ("y", "x")])
def test_multidimensional_self_brackets_preserve_ordered_axis(
    axis: tuple[str, ...],
) -> None:
    components = (
        _function(axis, ((Fraction(1), (33, 0)),)),
        _function(axis, ((Fraction(1), (0, 33)),)),
    )
    vector = RationalCoordinateTensor(
        coordinate_axis=axis, variance=("CONTRAVARIANT",), components=components
    )
    result = _assert_bracket(vector, vector)
    assert all(not value.numerator.terms for value in result.lie_derivative.components)


@pytest.mark.parametrize("height", [False, True])
def test_genuine_surviving_growth_remains_a_resource_refusal(height: bool) -> None:
    vector, tensor = (
        (_vector(33, 10**127), _vector(32, 10**127))
        if height
        else (_vector(128), _vector(127))
    )
    with pytest.raises(OperationResourceAdmissionError):
        lie_derivative(vector, tensor)


def test_unreduced_source_is_not_accepted_by_zero_presolve() -> None:
    value = _function(("x",), ((Fraction(1), (34,)),), (1,))
    vector = RationalCoordinateTensor(
        coordinate_axis=("x",),
        variance=("CONTRAVARIANT",),
        components=(value,),
        retained_nonzero_denominators=(value.denominator,),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        lie_derivative(vector, vector)
    assert type(error.value) is OperationDomainValidationError
    assert error.value.errors()[0]["type"].endswith("component_not_canonical")


@pytest.mark.parametrize(
    "malformation", ["exponent", "coefficient", "unreduced", "missing_guard"]
)
def test_zero_refinement_does_not_admit_forged_native_carriers(
    malformation: str,
) -> None:
    vector = (
        _vector(64, reciprocal=True)
        if malformation == "missing_guard"
        else _vector(128)
    )
    if malformation == "missing_guard":
        vector = vector.model_copy(update={"retained_nonzero_denominators": ()})
    else:
        source = vector.components[0]
        term = source.numerator.terms[0]
        if malformation == "exponent":
            term = term.model_copy(update={"exponents": (129,)})
        else:
            coefficient = (
                CanonicalRational.model_construct(num=2, den=2)
                if malformation == "unreduced"
                else CanonicalRational(num=10**128, den=1)
            )
            term = term.model_copy(update={"coefficient": coefficient})
        source = source.model_copy(
            update={
                "numerator": SparseRationalPolynomial.model_construct(terms=(term,))
            }
        )
        vector = vector.model_copy(update={"components": (source,)})
    with pytest.raises(OperationDomainValidationError) as error:
        lie_derivative(vector, vector)
    assert type(error.value) is OperationDomainValidationError


def test_refined_plan_retains_raw_normalization_and_exact_work_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _vector(33)
    plan = bounds.build_lie_derivative_plan(vector, vector)
    assert plan.components[0].raw_result.is_zero
    assert dict(plan.work_units_by_category)["normalization"] > 1
    monkeypatch.setattr(bounds, "MAX_LIE_DERIVATIVE_WORK_UNITS", plan.work_units)
    result = lie_derivative(vector, vector)
    assert verify_lie_derivative(result)
    monkeypatch.setattr(bounds, "MAX_LIE_DERIVATIVE_WORK_UNITS", plan.work_units - 1)
    with pytest.raises(OperationResourceAdmissionError):
        verify_lie_derivative(result)


def test_nonmonomial_rational_proportional_sources_keep_coprimality_and_locus() -> None:
    axis = ("x",)
    denominator = _function(axis, ((Fraction(1), (1,)), (Fraction(1), (0,)))).numerator
    components = tuple(
        _function(axis, ((Fraction(scale), (34,)), (Fraction(scale), (0,)))).model_copy(
            update={"denominator": denominator}
        )
        for scale in (1, 7)
    )
    vectors = tuple(
        RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=("CONTRAVARIANT",),
            components=(component,),
            retained_nonzero_denominators=(denominator,),
        )
        for component in components
    )
    result = _assert_bracket(vectors[0], vectors[1])
    assert not result.lie_derivative.components[0].numerator.terms
    assert result.lie_derivative.retained_nonzero_denominators == (denominator,)


@pytest.mark.parametrize(
    "missing", ["axis", "numerator", "denominator", "coefficient", "zero_denominator"]
)
def test_malformed_native_source_is_typed_before_bound_arithmetic(missing: str) -> None:
    vector = _vector(128)
    component = vector.components[0]
    if missing == "axis":
        vector = RationalCoordinateTensor.model_construct(
            variance=vector.variance,
            components=vector.components,
            retained_nonzero_denominators=(),
        )
    elif missing == "numerator":
        component = RationalFunction.model_construct(
            variables=("x",), denominator=component.denominator
        )
    elif missing == "denominator":
        component = RationalFunction.model_construct(
            variables=("x",), numerator=component.numerator
        )
    else:
        term = (
            RationalPolynomialTerm.model_construct(exponents=(128,))
            if missing == "coefficient"
            else component.numerator.terms[0].model_copy(
                update={"coefficient": CanonicalRational.model_construct(num=1, den=0)}
            )
        )
        component = component.model_copy(
            update={
                "numerator": SparseRationalPolynomial.model_construct(terms=(term,))
            }
        )
    if missing != "axis":
        vector = vector.model_copy(update={"components": (component,)})
    with pytest.raises(OperationDomainValidationError) as error:
        lie_derivative(vector, vector)
    assert type(error.value) is OperationDomainValidationError
