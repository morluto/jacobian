"""Bounded exact polynomial orbits under checked additive-group coactions."""

from fractions import Fraction

import pytest
from tests.error_assertions import error_code

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.derivations._models import PolynomialGaAction
from jacobian.math.polynomials.derivations.orbits._kernels import (
    ga_polynomial_orbit,
)
from jacobian.math.polynomials.derivations.orbits._models import (
    GaPolynomialOrbitRequest,
)
from jacobian.math.polynomials.derivations.orbits._tools import TOOLS
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int | Fraction]
) -> RationalPolynomial:
    canonical = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(terms.items(), reverse=True)
        if coefficient
    )
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(terms=canonical),
    )


def _translation_action() -> PolynomialGaAction:
    return PolynomialGaAction(
        source_variables=("x", "y"),
        parameter="t",
        generator_images=(
            _polynomial(
                ("x", "y", "t"),
                {(1, 0, 0): 1, (0, 1, 1): 1},
            ),
            _polynomial(("x", "y", "t"), {(0, 1, 0): 1}),
        ),
    )


def _terms(value: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in value.polynomial.terms
    }


def test_translation_orbit_matches_direct_polynomial_expansion() -> None:
    action = _translation_action()
    source = _polynomial(
        ("x", "y"),
        {(2, 0): 1, (1, 1): 3, (0, 2): 1},
    )

    result = ga_polynomial_orbit(action, source)

    assert result.variables == ("x", "y", "t")
    assert _terms(result) == {
        (2, 0, 0): Fraction(1),
        (1, 1, 1): Fraction(2),
        (0, 2, 2): Fraction(1),
        (1, 1, 0): Fraction(3),
        (0, 2, 1): Fraction(3),
        (0, 2, 0): Fraction(1),
    }


def test_invariant_and_zero_polynomials_retain_extended_parent() -> None:
    action = _translation_action()

    invariant = ga_polynomial_orbit(
        action, _polynomial(("x", "y"), {(0, 1): 7, (0, 0): 2})
    )
    zero = ga_polynomial_orbit(action, _polynomial(("x", "y"), {}))

    assert invariant.variables == ("x", "y", "t")
    assert _terms(invariant) == {(0, 1, 0): Fraction(7), (0, 0, 0): Fraction(2)}
    assert zero.variables == ("x", "y", "t")
    assert zero.polynomial.terms == ()


def test_orbit_rejects_wrong_ordered_ring_and_invalid_action() -> None:
    action = _translation_action()
    with pytest.raises(ValueError) as exc_info:
        GaPolynomialOrbitRequest(
            action=action,
            polynomial=_polynomial(("y", "x"), {(1, 0): 1}),
        )
    assert error_code(exc_info.value) == "polynomial_ga_orbit.ordered_ring_mismatch"

    invalid = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_polynomial(("x", "t"), {(1, 0): 1, (0, 2): 1}),),
    )
    with pytest.raises(OperationDomainValidationError):
        ga_polynomial_orbit(invalid, _polynomial(("x",), {(1,): 1}))


def test_invariant_many_terms_bound_denominators_per_output_collision() -> None:
    action = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_polynomial(("x", "t"), {(1, 0): 1}),),
    )
    source = _polynomial(
        ("x",),
        {(degree,): Fraction(1, 11) for degree in range(65)},
    )

    result = ga_polynomial_orbit(action, source)

    assert len(result.polynomial.terms) == 65
    assert all(
        term.coefficient.as_fraction() == Fraction(1, 11)
        for term in result.polynomial.terms
    )


def test_orbit_bounds_repeated_image_denominator_before_expansion() -> None:
    q = 10**49 + 1
    action = PolynomialGaAction(
        source_variables=("x", "y"),
        parameter="t",
        generator_images=(
            _polynomial(
                ("x", "y", "t"),
                {(1, 0, 0): 1, (0, 1, 1): Fraction(1, q)},
            ),
            _polynomial(("x", "y", "t"), {(0, 1, 0): 1}),
        ),
    )
    source = _polynomial(("x", "y"), {(12, 0): 1})

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_polynomial_orbit(action, source)
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_ga_orbit.coefficient_growth"
    )


def test_orbit_admits_a_merged_support_within_the_envelope() -> None:
    """(x+t)**13 collapses to 14 distinct monomials, so it is admitted."""
    action = PolynomialGaAction(
        source_variables=("x",),
        parameter="t",
        generator_images=(_polynomial(("x", "t"), {(1, 0): 1, (0, 1): 1}),),
    )
    image = ga_polynomial_orbit(action, _polynomial(("x",), {(13,): 1}))
    assert len(image.polynomial.terms) == 14


def test_orbit_rejects_expansion_before_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.polynomials.derivations.orbits._kernels as operations

    # Each generator is x_i + t, so a source monomial of degree d expands to
    # the product of the two affine images and yields one distinct monomial per
    # exponent split. Four degree-64 monomials in two variables need far more
    # than the admitted 4,096 expansion envelope.
    action = PolynomialGaAction(
        source_variables=("x", "y"),
        parameter="t",
        generator_images=(
            _polynomial(("x", "y", "t"), {(0, 0, 1): 1, (1, 0, 0): 1}),
            _polynomial(("x", "y", "t"), {(0, 0, 1): 1, (0, 1, 0): 1}),
        ),
    )
    source = _polynomial(
        ("x", "y"),
        {(32, 32): 1, (31, 33): 1, (33, 31): 1, (30, 34): 1},
    )

    def expansion_must_not_start(*args: object, **kwargs: object) -> None:
        pytest.fail("orbit expansion started before resource admission")

    monkeypatch.setattr(operations, "_expand", expansion_must_not_start)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_polynomial_orbit(action, source)
    assert exc_info.value.errors()[0]["type"] == "polynomial_ga_orbit.expansion_budget"


def test_catalog_declaration_example_is_typed_and_computes_the_orbit() -> None:
    declaration = TOOLS[0]
    assert declaration.operation_id == "algebraic_group.ga.polynomial_orbit.compute"
    request = GaPolynomialOrbitRequest.model_validate_json(
        encode_strict_json(declaration.examples[0].input), strict=True
    )

    result = ga_polynomial_orbit(request.action, request.polynomial)

    assert _terms(result) == {
        (2, 0, 0): Fraction(1),
        (1, 1, 1): Fraction(2),
        (0, 2, 2): Fraction(1),
    }


def test_orbit_bounds_denominators_for_terms_with_overlapping_support() -> None:
    action = PolynomialGaAction(
        source_variables=("x", "y", "z"),
        parameter="t",
        generator_images=(
            _polynomial(("x", "y", "z", "t"), {(1, 0, 0, 0): 1, (0, 0, 1, 1): 1}),
            _polynomial(("x", "y", "z", "t"), {(0, 1, 0, 0): 1, (0, 0, 1, 1): 1}),
            _polynomial(("x", "y", "z", "t"), {(0, 0, 1, 0): 1}),
        ),
    )
    p, q = 10**99 + 1, 10**99 + 3
    assert p != q and Fraction(1, p) + Fraction(1, q) == Fraction(p + q, p * q)
    source = _polynomial(
        ("x", "y", "z"), {(1, 0, 0): Fraction(1, p), (0, 1, 0): Fraction(1, q)}
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        ga_polynomial_orbit(action, source)
    assert (
        exc_info.value.errors()[0]["type"] == "polynomial_ga_orbit.coefficient_growth"
    )
