"""Exact splitting-field binding for root-critical distances (#3722)."""

from __future__ import annotations

from fractions import Fraction

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
)


def _poly(*ascending: int):
    return _polynomial_from_ascending([Fraction(value) for value in ascending], "x")


def _fractions(coefficients) -> tuple[Fraction, ...]:
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
        exact_splitting_field("not a polynomial")
