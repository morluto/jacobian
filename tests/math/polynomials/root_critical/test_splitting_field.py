"""Exact splitting-field binding for root-critical distances (#3722)."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.root_critical import (
    ExactSplittingField,
    exact_splitting_field,
    splitting_field_distance_profile,
)
from jacobian.math.polynomials.root_critical._models import (
    SplittingFieldDistanceRequest,
)
from jacobian.math.polynomials.root_critical._splitting import (
    _polynomial_from_ascending,
    conjugate_element,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(*ascending: int) -> RationalPolynomial:
    return _polynomial_from_ascending([Fraction(value) for value in ascending], "x")


def _fractions(coefficients: Iterable[Any]) -> tuple[Fraction, ...]:
    return tuple(value.as_fraction() for value in coefficients)


def test_quadratic_splitting_field_known_answer() -> None:
    field = exact_splitting_field(_poly(-2, 0, 1))
    # The support of (x^2-2) * (2x) is x^3 - 2x, whose splitting field is Q(sqrt 2).
    assert field.squarefree_support == _poly(0, -2, 0, 1)
    expected_defining = _polynomial_from_ascending(
        [Fraction(-2), Fraction(0), Fraction(1)], "t"
    )
    assert field.defining_polynomial == expected_defining
    # theta = sqrt(2) or -sqrt(2): conjugation fixes it.
    assert _fractions(field.conjugation_coefficients) in (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
    )
    root_coefficients = {
        _fractions(root.coefficients_ascending) for root in field.roots
    }
    assert root_coefficients == {
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
        (Fraction(0), Fraction(0)),
    }


def test_bound_distance_quadratic_is_two() -> None:
    polynomial = _poly(-2, 0, 1)
    field = exact_splitting_field(polynomial)
    result = splitting_field_distance_profile(polynomial, field)
    assert len(result.profile.pairs) == 2
    for row, coefficients in zip(
        result.profile.pairs, result.distance_field_coefficients, strict=True
    ):
        assert row.kind == "POSITIVE"
        assert _fractions(coefficients) == (Fraction(2), Fraction(0))
        assert row.isolating_interval.lower.as_fraction() >= 0


def test_bound_distance_golden_ratio() -> None:
    """(phi - 1/2)^2 = 5/4 exactly, and 5/4 is rational in the field."""
    polynomial = _poly(-1, -1, 1)
    field = exact_splitting_field(polynomial)
    result = splitting_field_distance_profile(polynomial, field)
    assert len(result.profile.pairs) == 2
    for coefficients in result.distance_field_coefficients:
        assert _fractions(coefficients) == (Fraction(5, 4), Fraction(0))


def test_cubic_splitting_field_degree_and_conjugation() -> None:
    """x^3-2 has a degree-six splitting field with conjugation theta -> -theta."""
    polynomial = _poly(-2, 0, 0, 1)
    field = exact_splitting_field(polynomial)
    degree = field.defining_polynomial.polynomial.terms[0].exponents[0]
    assert degree == 6
    conjugation = _fractions(field.conjugation_coefficients)
    assert conjugation == (Fraction(0), Fraction(-1), *([Fraction(0)] * 4))
    # The support of x^3-2 times 3x^2 is square-free and contains 0.
    assert len(field.roots) == 4
    zero_root = next(
        root
        for root in field.roots
        if _fractions(root.coefficients_ascending)
        == (Fraction(0),) + (Fraction(0),) * 5
    )
    assert zero_root.axis_index == 0


def test_zero_distance_row_has_zero_field_element() -> None:
    polynomial = _poly(0, 0, 1)  # x^2 double root at 0
    field = exact_splitting_field(polynomial)
    result = splitting_field_distance_profile(polynomial, field)
    assert len(result.profile.pairs) == 1
    row = result.profile.pairs[0]
    assert row.kind == "ZERO_DISTANCE"
    assert _fractions(result.distance_field_coefficients[0]) == (Fraction(0),)


def test_unbound_field_is_rejected() -> None:
    field = exact_splitting_field(_poly(-2, 0, 1))
    with pytest.raises(OperationDomainValidationError, match="bound to the exact"):
        splitting_field_distance_profile(_poly(-1, 0, 1), field)


def test_forged_defining_polynomial_is_rejected() -> None:
    polynomial = _poly(-2, 0, 1)
    field = exact_splitting_field(polynomial)
    forged = field.model_copy(update={"defining_polynomial": _poly(-1, 0, 1)})
    with pytest.raises(OperationDomainValidationError, match="computed exact field"):
        splitting_field_distance_profile(polynomial, forged)


def test_canonical_rational_round_trip() -> None:
    polynomial = _poly(-2, 0, 1)
    field = exact_splitting_field(polynomial)
    restored = ExactSplittingField.model_validate_json(field.model_dump_json())
    assert restored == field
    result = splitting_field_distance_profile(polynomial, field)
    restored_result = type(result).model_validate_json(result.model_dump_json())
    assert restored_result == result


def test_request_contract_is_structural() -> None:
    """Request parsing is structural; field binding is enforced by the kernel."""

    field = exact_splitting_field(_poly(-2, 0, 1))
    request = SplittingFieldDistanceRequest(
        polynomial=_poly(-2, 0, 1),
        splitting_field=field,
        max_pair_rows=4,
    )
    assert request.splitting_field is field
    assert request.max_pair_rows == 4


def test_field_degree_bound_rejected() -> None:
    # x^19 - 2 has degree 19 support; rejected before any backend work.
    coefficients = [Fraction(0)] * 19 + [Fraction(-2), Fraction(1)]
    with pytest.raises(OperationDomainValidationError):
        exact_splitting_field(_polynomial_from_ascending(coefficients, "x"))


def test_native_field_type_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="rational polynomial"):
        exact_splitting_field("not a polynomial")  # type: ignore[arg-type]


def _sympy_quotient_check(field: ExactSplittingField) -> None:
    """Every retained root must annihilate the support in Q[t]/(defining)."""
    from sympy import Poly, Rational, Symbol

    t = Symbol("t")
    modulus = Poly(
        [
            Rational(c.numerator, c.denominator)
            for c in reversed(_ascending(field.defining_polynomial))
        ],
        t,
        domain="QQ",
    )
    assert modulus.is_monic
    assert modulus.is_irreducible
    support = _ascending(field.squarefree_support)
    for root in field.roots:
        value = Poly(
            [
                Rational(c.numerator, c.denominator)
                for c in reversed(_fractions(root.coefficients_ascending))
            ],
            t,
            domain="QQ",
        )
        evaluated = Poly(
            sum(coeff * value**power for power, coeff in enumerate(support)),
            t,
            domain="QQ",
        )
        assert evaluated.rem(modulus).is_zero


def _ascending(polynomial: RationalPolynomial) -> list[Fraction]:
    terms = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }
    degree = max(terms) if terms else 0
    return [terms.get(power, Fraction(0)) for power in range(degree + 1)]


def test_roots_annihilate_support_in_quotient_field() -> None:
    _sympy_quotient_check(exact_splitting_field(_poly(-2, 0, 1)))
    _sympy_quotient_check(exact_splitting_field(_poly(-2, 0, 0, 1)))
    _sympy_quotient_check(exact_splitting_field(_poly(0, 0, 1)))


def test_forged_root_fails_quotient_reconstruction() -> None:
    from sympy import Poly, Rational, Symbol

    field = exact_splitting_field(_poly(-2, 0, 1))
    t = Symbol("t")
    modulus = Poly(
        [
            Rational(c.numerator, c.denominator)
            for c in reversed(_ascending(field.defining_polynomial))
        ],
        t,
        domain="QQ",
    )
    support = _ascending(field.squarefree_support)
    first = _fractions(field.roots[0].coefficients_ascending)
    forged = [first[0] + Fraction(1), *first[1:]]
    value = Poly(
        [Rational(c.numerator, c.denominator) for c in reversed(forged)],
        t,
        domain="QQ",
    )
    evaluated = Poly(
        sum(coeff * value**power for power, coeff in enumerate(support)),
        t,
        domain="QQ",
    )
    assert not evaluated.rem(modulus).is_zero


def test_conjugation_is_a_nontrivial_involution() -> None:
    field = exact_splitting_field(_poly(-2, 0, 0, 1))
    modulus = _ascending(field.defining_polynomial)
    conjugation = list(_fractions(field.conjugation_coefficients))
    degree = len(modulus) - 1
    # Conjugation fixes every rational constant.
    assert conjugate_element(
        [Fraction(5)] + [Fraction(0)] * (degree - 1), conjugation, modulus
    ) == [Fraction(5)] + [Fraction(0)] * (degree - 1)
    # Applying it twice is the identity on the power basis.
    for power in range(degree):
        basis = [Fraction(0)] * degree
        basis[power] = Fraction(1)
        assert (
            conjugate_element(
                conjugate_element(basis, conjugation, modulus), conjugation, modulus
            )
            == basis
        )
    # It is nontrivial: some root moves under conjugation.
    roots = [list(_fractions(root.coefficients_ascending)) for root in field.roots]
    assert any(conjugate_element(root, conjugation, modulus) != root for root in roots)
    # It permutes the retained root family.
    assert {tuple(conjugate_element(root, conjugation, modulus)) for root in roots} == {
        tuple(root) for root in roots
    }


def test_forged_identity_conjugation_misses_root_motion() -> None:
    """The identity map is an automorphism but not complex conjugation here."""
    field = exact_splitting_field(_poly(-2, 0, 0, 1))
    modulus = _ascending(field.defining_polynomial)
    degree = len(modulus) - 1
    true_conjugation = list(_fractions(field.conjugation_coefficients))
    identity_map = [Fraction(0), Fraction(1)] + [Fraction(0)] * (degree - 2)
    roots = [list(_fractions(root.coefficients_ascending)) for root in field.roots]
    assert any(
        conjugate_element(root, true_conjugation, modulus) != root for root in roots
    )
    assert all(conjugate_element(root, identity_map, modulus) == root for root in roots)


def test_root_rectangles_are_pairwise_distinct() -> None:
    field = exact_splitting_field(_poly(-2, 0, 0, 1))
    boxes = [
        (
            root.rectangle.real_lower.as_fraction(),
            root.rectangle.real_upper.as_fraction(),
            root.rectangle.imaginary_lower.as_fraction(),
            root.rectangle.imaginary_upper.as_fraction(),
        )
        for root in field.roots
    ]
    assert len(set(boxes)) == len(boxes)
    for real_lower, real_upper, imag_lower, imag_upper in boxes:
        assert real_lower <= real_upper
        assert imag_lower <= imag_upper
