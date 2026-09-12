"""Canonical elementary symmetric families."""

from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Poly, expand, prod, symbols

from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy
from jacobian.math.polynomials._elementary_symmetric import (
    ElementarySymmetricFamilyRequest,
    elementary_symmetric_family,
)
from jacobian.math.polynomials._elementary_symmetric_tools import (
    ELEMENTARY_SYMMETRIC_FAMILY_OPERATION,
)
from jacobian.math.polynomials.values import RationalPolynomial


def support(polynomial: RationalPolynomial) -> tuple[tuple[int, ...], ...]:
    return tuple(term.exponents for term in polynomial.polynomial.terms)


def coefficients(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def test_family_through_degree_two() -> None:
    result = elementary_symmetric_family(
        ("x", "y", "z"),
        2,
    )
    assert support(result.polynomials[0]) == ((0, 0, 0),)
    assert support(result.polynomials[1]) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert support(result.polynomials[2]) == ((1, 1, 0), (1, 0, 1), (0, 1, 1))


def test_empty_axis_e_zero_is_a_canonical_constant() -> None:
    result = elementary_symmetric_family(
        (),
        0,
    )
    assert result.polynomials[0].variables == ()
    assert support(result.polynomials[0]) == ((),)


def test_request_rejects_duplicate_variables_and_degree_above_axis() -> None:
    with pytest.raises(ValidationError, match="variables must be unique"):
        ElementarySymmetricFamilyRequest(variables=("x", "x"), maximum_degree=1)
    with pytest.raises(ValidationError, match="cannot exceed the variable count"):
        ElementarySymmetricFamilyRequest(variables=("x",), maximum_degree=2)


def test_dynamic_recurrence_matches_prefix_recurrence() -> None:
    full = elementary_symmetric_family(
        ("x", "y", "z"),
        3,
    )
    prefix = elementary_symmetric_family(
        ("x", "y"),
        2,
    )
    for degree in range(1, 4):
        expected = (
            {
                (*exponents, 0): value
                for exponents, value in coefficients(prefix.polynomials[degree]).items()
            }
            if degree <= 2
            else {}
        )
        if degree:
            for exponents, value in coefficients(
                prefix.polynomials[degree - 1]
            ).items():
                shifted = (*exponents, 1)
                expected[shifted] = expected.get(shifted, Fraction()) + value
        assert coefficients(full.polynomials[degree]) == expected


def test_family_is_invariant_under_coherent_variable_permutation() -> None:
    ordered = elementary_symmetric_family(
        ("x", "y", "z"),
        3,
    )
    permuted = elementary_symmetric_family(
        ("z", "x", "y"),
        3,
    )
    for left, right in zip(ordered.polynomials, permuted.polynomials, strict=True):
        left_by_name = {
            tuple(sorted(zip(left.variables, exponents, strict=True))): value
            for exponents, value in coefficients(left).items()
        }
        right_by_name = {
            tuple(sorted(zip(right.variables, exponents, strict=True))): value
            for exponents, value in coefficients(right).items()
        }
        assert left_by_name == right_by_name


def test_vieta_reconstruction_matches_product_of_linear_factors() -> None:
    variables = ("x", "y", "z")
    result = elementary_symmetric_family(
        variables,
        3,
    )
    t = symbols("t")
    x, y, z = symbols("x y z")
    expected = Poly(prod(t - variable for variable in (x, y, z)).expand(), t)
    for degree, polynomial in enumerate(result.polynomials):
        actual = rational_polynomial_to_sympy(polynomial).as_expr()
        coefficient = expected.coeff_monomial(t ** (3 - degree))
        assert expand(actual - (-1) ** degree * coefficient) == 0


def test_complete_eighth_axis_family_stays_within_exact_carrier() -> None:
    result = elementary_symmetric_family(
        tuple(f"x{index}" for index in range(8)),
        8,
    )
    assert len(result.polynomials) == 9
    assert (
        sum(len(polynomial.polynomial.terms) for polynomial in result.polynomials)
        == 256
    )


def test_catalog_declaration_uses_requested_operation_id_and_round_trips() -> None:
    assert (
        ELEMENTARY_SYMMETRIC_FAMILY_OPERATION.operation_id
        == "polynomial.symmetric.elementary_family.compute"
    )
    request = ElementarySymmetricFamilyRequest(
        variables=("x", "y", "z"), maximum_degree=2
    )
    result = ELEMENTARY_SYMMETRIC_FAMILY_OPERATION.run(request)
    decoded = ELEMENTARY_SYMMETRIC_FAMILY_OPERATION.result_type.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result


def test_result_rejects_forged_nonunit_e_zero_without_replaying_the_family() -> None:
    result = elementary_symmetric_family(
        ("x", "y"),
        1,
    )
    forged = result.model_dump()
    forged["polynomials"][0]["polynomial"]["terms"][0]["coefficient"]["num"] = 2
    with pytest.raises(ValidationError, match="canonical constant one"):
        type(result).model_validate(forged)
