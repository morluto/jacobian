"""Colliding translated source fractions are admitted before substitution."""

from fractions import Fraction

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.derivations import _stable_kernels
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.derivations._stable_kernels import (
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _polynomial(
    variables: tuple[str, ...], terms: tuple[tuple[tuple[int, ...], Fraction], ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {
                        "exponents": exponent,
                        "coefficient": {
                            "num": coefficient.numerator,
                            "den": coefficient.denominator,
                        },
                    }
                    for exponent, coefficient in sorted(terms, reverse=True)
                    if coefficient
                ]
            },
        }
    )


def _translation(
    coefficients: tuple[Fraction, ...],
) -> tuple[PolynomialGaAction, tuple[RationalPolynomial, ...]]:
    variables = tuple(f"x{index}" for index in range(len(coefficients)))
    axes = tuple(
        tuple(int(i == j) for j in range(len(variables))) for i in range(len(variables))
    )
    action = PolynomialGaAction(
        source_variables=variables,
        parameter="t",
        generator_images=tuple(
            _polynomial(
                (*variables, "t"),
                (
                    ((*axis, 0), Fraction(1)),
                    ((*(0 for _ in variables), 1), Fraction(1)),
                ),
            )
            for axis in axes
        ),
    )
    basis = (
        _polynomial(variables, (((0,) * len(variables), Fraction(1)),)),
        _polynomial(variables, tuple(zip(axes, coefficients, strict=True))),
    )
    return action, basis


@pytest.mark.parametrize("public", (False, True))
def test_distinct_source_denominators_are_refused_before_expansion(
    public: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pairwise coprime 50-digit prime powers yield a 150-digit t denominator.
    coefficients = (
        Fraction(1),
        Fraction(1, 2**163),
        Fraction(1, 3**103),
        Fraction(1, 5**71),
    )
    assert len(str(sum(coefficients).denominator)) > 128
    action, basis = _translation(coefficients)

    def unexpected_expansion(*_args: object, **_kwargs: object) -> object:
        pytest.fail("source denominator growth must be refused before substitution")

    monkeypatch.setattr(_stable_kernels, "_substitute_basis", unexpected_expansion)
    with pytest.raises(OperationResourceAdmissionError) as error:
        if public:
            invoke_operation(
                "algebraic_group.ga.stable_subrepresentation.compute",
                {
                    "action": action.model_dump(mode="json"),
                    "basis": [value.model_dump(mode="json") for value in basis],
                },
                Catalog.open(),
            )
        else:
            ga_stable_subrepresentation(action, basis)
    assert (
        error.value.errors()[0]["type"]
        == "polynomial_ga_subrepresentation.coefficient_growth"
    )


@pytest.mark.parametrize(
    "coefficients",
    (
        (
            Fraction(1),
            Fraction(1, 10**49 + 1),
            Fraction(2, 10**49 + 1),
            Fraction(3, 10**49 + 1),
        ),
        (Fraction(1), Fraction(1, 10**49 + 1), Fraction(-1, 10**49 + 1)),
        (Fraction(1), Fraction(1, 2**30), Fraction(1, 3**18), Fraction(1, 5**13)),
    ),
    ids=("repeated-denominator", "cancelled-fractions", "distinct-small-denominators"),
)
def test_shared_source_denominators_preserve_native_public_exact_matrix(
    coefficients: tuple[Fraction, ...],
) -> None:
    action, basis = _translation(coefficients)
    result = ga_stable_subrepresentation(action, basis)
    expected = sum(coefficients)
    assert {
        term.exponents: term.coefficient.as_fraction()
        for term in result.action_matrix[0][1].polynomial.terms
    } == {(1,): expected}
    public = invoke_operation(
        "algebraic_group.ga.stable_subrepresentation.compute",
        {
            "action": action.model_dump(mode="json"),
            "basis": [value.model_dump(mode="json") for value in basis],
        },
        Catalog.open(),
    )
    assert public.output == result.model_dump(mode="json")


@pytest.mark.parametrize("public", (False, True))
def test_zero_basis_polynomial_keeps_the_dependence_diagnostic(public: bool) -> None:
    action, basis = _translation((Fraction(0),))
    with pytest.raises(OperationDomainValidationError) as error:
        if public:
            invoke_operation(
                "algebraic_group.ga.stable_subrepresentation.compute",
                {
                    "action": action.model_dump(mode="json"),
                    "basis": [value.model_dump(mode="json") for value in basis],
                },
                Catalog.open(),
            )
        else:
            ga_stable_subrepresentation(action, basis)
    assert error.value.errors()[0]["type"] == (
        "polynomial_ga_subrepresentation.dependent_basis"
    )
