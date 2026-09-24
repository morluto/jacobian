"""Exact Puiseux arithmetic with independent sparse coefficient oracles."""

from fractions import Fraction
from math import gcd

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series import (
    PuiseuxTerm,
    TruncatedPuiseuxWindow,
    add_puiseux,
    differentiate_puiseux,
    inverse_puiseux,
    multiply_puiseux,
    residue_puiseux,
    subtract_puiseux,
)


def q(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def window(
    terms: tuple[tuple[Fraction, int], ...],
    *,
    lower: Fraction = Fraction(0),
    precision: Fraction = Fraction(3),
    center: Fraction = Fraction(0),
    variable: str = "t",
) -> TruncatedPuiseuxWindow:
    fractions = (lower, precision, center, *(exponent for exponent, _ in terms))
    ramification = 1
    for value in fractions:
        ramification = (
            ramification * value.denominator // gcd(ramification, value.denominator)
        )
    return TruncatedPuiseuxWindow(
        variable=variable,
        center=CanonicalRational.from_fraction(center),
        valuation_lower=CanonicalRational.from_fraction(lower),
        precision=CanonicalRational.from_fraction(precision),
        ramification_index=ramification,
        terms=tuple(
            PuiseuxTerm(
                exponent=CanonicalRational.from_fraction(exponent),
                coefficient=q(coefficient),
            )
            for exponent, coefficient in terms
        ),
    )


def coefficients(value: TruncatedPuiseuxWindow) -> dict[Fraction, Fraction]:
    return {
        term.exponent.as_fraction(): term.coefficient.as_fraction()
        for term in value.terms
    }


def test_puiseux_residue_uses_exact_t_minus_one_coefficient() -> None:
    source = window(
        ((Fraction(-3, 2), 7), (Fraction(-1), 5), (Fraction(1, 2), 9)),
        lower=Fraction(-2),
        precision=Fraction(2),
    )

    result = residue_puiseux(source)

    assert result.series == source
    assert result.residue.as_fraction() == Fraction(5)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_puiseux_residue_returns_zero_only_when_minus_one_is_known_absent() -> None:
    source = window((), lower=Fraction(-2), precision=Fraction(0))

    assert residue_puiseux(source).residue.as_fraction() == 0


@pytest.mark.parametrize(
    ("lower", "precision"),
    [(Fraction(0), Fraction(2)), (Fraction(-2), Fraction(-1))],
)
def test_puiseux_residue_rejects_windows_that_do_not_determine_minus_one(
    lower: Fraction, precision: Fraction
) -> None:
    source = window((), lower=lower, precision=precision)

    with pytest.raises(
        OperationDomainValidationError, match="must contain exponent -1"
    ):
        residue_puiseux(source)


def test_add_combines_different_ramification_lattices_exactly() -> None:
    half = window(((Fraction(1, 2), 2),))
    third = window(((Fraction(1, 3), 5),))

    result = add_puiseux(half, third)

    assert result.ramification_index == 6
    assert coefficients(result) == {
        Fraction(1, 3): Fraction(5),
        Fraction(1, 2): Fraction(2),
    }
    restored = TruncatedPuiseuxWindow.model_validate_json(result.model_dump_json())
    assert restored == result


def test_subtraction_cancels_to_a_finite_zero_prefix() -> None:
    source = window(
        ((Fraction(-1, 2), 7), (Fraction(2, 3), -3)),
        lower=Fraction(-1, 2),
    )

    result = subtract_puiseux(source, source)

    assert result.terms == ()
    assert result.valuation_lower == source.valuation_lower
    assert result.precision == source.precision


def test_product_matches_independent_cauchy_product_and_safe_tail_cutoff() -> None:
    left = window(
        ((Fraction(1, 2), 2), (Fraction(1), 3)),
        precision=Fraction(3, 2),
    )
    right = window(
        ((Fraction(1, 3), 5), (Fraction(1), -1)),
        precision=Fraction(2),
    )

    result = multiply_puiseux(left, right)

    # Independently enumerate exact pairs below min(P_left+v_right,
    # P_right+v_left)=min(11/6, 5/2)=11/6.
    oracle: dict[Fraction, Fraction] = {}
    for a, ca in ((Fraction(1, 2), 2), (Fraction(1), 3)):
        for b, cb in ((Fraction(1, 3), 5), (Fraction(1), -1)):
            exponent = a + b
            if exponent < Fraction(11, 6):
                oracle[exponent] = oracle.get(exponent, Fraction(0)) + ca * cb
    assert result.precision.as_fraction() == Fraction(11, 6)
    assert result.ramification_index == 6
    assert coefficients(result) == oracle
    assert all(exponent < Fraction(11, 6) for exponent in coefficients(result))


def test_derivative_matches_exact_monomial_rule_and_transports_cutoff() -> None:
    source = window(
        ((Fraction(-1, 2), 6), (Fraction(0), 11), (Fraction(2, 3), -9)),
        lower=Fraction(-1, 2),
        precision=Fraction(5, 3),
        center=Fraction(2),
    )

    result = differentiate_puiseux(source)

    # Independent termwise oracle: d(c*t^q)/dt = q*c*t^(q-1).
    assert result.center == source.center
    assert result.variable == source.variable
    assert result.valuation_lower.as_fraction() == Fraction(-3, 2)
    assert result.precision.as_fraction() == Fraction(2, 3)
    assert result.ramification_index == 6
    assert coefficients(result) == {
        Fraction(-3, 2): Fraction(-3),
        Fraction(-1, 3): Fraction(-6),
    }
    assert (
        TruncatedPuiseuxWindow.model_validate_json(result.model_dump_json()) == result
    )


def test_derivative_preserves_zero_prefix_without_claiming_global_zero() -> None:
    source = window((), lower=Fraction(-1, 3), precision=Fraction(4, 3))

    result = differentiate_puiseux(source)

    assert result.terms == ()
    assert result.valuation_lower.as_fraction() == Fraction(-4, 3)
    assert result.precision.as_fraction() == Fraction(1, 3)


def test_derivative_rejects_coefficient_growth_before_multiplication() -> None:
    source = window(
        ((Fraction(999_999), 10**4_095),),
        precision=Fraction(1_000_000),
    )

    with pytest.raises(OperationResourceAdmissionError, match="coefficient growth"):
        differentiate_puiseux(source)


def test_inverse_uses_exact_unit_recurrence_and_justified_precision() -> None:
    source = window(
        (
            (Fraction(1, 2), 1),
            (Fraction(1), 2),
            (Fraction(3, 2), 3),
        ),
        precision=Fraction(2),
    )

    result = inverse_puiseux(source)

    # In s=t^(1/2), (1+2s+3s^2)^-1 = 1-2s+s^2+O(s^3).
    assert result.valuation_lower.as_fraction() == Fraction(-1, 2)
    assert result.precision.as_fraction() == 1
    assert result.ramification_index == 2
    assert coefficients(result) == {
        Fraction(-1, 2): Fraction(1),
        Fraction(0): Fraction(-2),
        Fraction(1, 2): Fraction(1),
    }
    product = multiply_puiseux(source, result)
    assert product.precision.as_fraction() == Fraction(3, 2)
    assert coefficients(product) == {Fraction(0): Fraction(1)}
    assert (
        TruncatedPuiseuxWindow.model_validate_json(result.model_dump_json()) == result
    )


def test_inverse_normalizes_rational_coefficients_and_negative_valuation() -> None:
    source = TruncatedPuiseuxWindow(
        valuation_lower=q(-1, 3),
        precision=q(2, 3),
        ramification_index=3,
        terms=(
            PuiseuxTerm(exponent=q(-1, 3), coefficient=q(2)),
            PuiseuxTerm(exponent=q(0), coefficient=q(3, 2)),
        ),
    )

    result = inverse_puiseux(source)

    assert result.valuation_lower.as_fraction() == Fraction(1, 3)
    assert result.precision.as_fraction() == Fraction(4, 3)
    assert coefficients(result) == {
        Fraction(1, 3): Fraction(1, 2),
        Fraction(2, 3): Fraction(-3, 8),
        Fraction(1): Fraction(9, 32),
    }


def test_inverse_of_monomial_stays_sparse_at_large_known_precision() -> None:
    source = window(
        ((Fraction(2, 3), 5),),
        lower=Fraction(-1),
        precision=Fraction(1_000),
    )

    result = inverse_puiseux(source)

    assert result.valuation_lower.as_fraction() == Fraction(-2, 3)
    assert result.precision.as_fraction() == Fraction(2_996, 3)
    assert coefficients(result) == {Fraction(-2, 3): Fraction(1, 5)}


def test_inverse_does_not_treat_empty_finite_prefix_as_noninvertible_series() -> None:
    source = window((), lower=Fraction(0), precision=Fraction(2))

    with pytest.raises(OperationDomainValidationError, match="unknown nonzero tail"):
        inverse_puiseux(source)


def test_inverse_preflights_recurrence_coefficient_growth() -> None:
    source = window(
        ((Fraction(0), 1), (Fraction(1), 10**4_095)),
        precision=Fraction(3),
    )

    with pytest.raises(OperationResourceAdmissionError, match="recurrence may exceed"):
        inverse_puiseux(source)


def test_inverse_preflights_dense_term_recurrence_work() -> None:
    source = window(
        tuple((Fraction(index), 1) for index in range(301)),
        precision=Fraction(4_096),
    )

    with pytest.raises(OperationResourceAdmissionError, match="recurrence work"):
        inverse_puiseux(source)


def test_zero_factor_uses_known_zero_prefix_for_product_precision() -> None:
    zero_prefix = window((), lower=Fraction(-1, 2), precision=Fraction(1, 2))
    factor = window(((Fraction(1, 3), 2),), precision=Fraction(2))

    result = multiply_puiseux(zero_prefix, factor)

    assert result.terms == ()
    # The zero prefix's omitted tail begins at 1/2, so the product is known
    # zero only through 1/2 + 1/3.
    assert result.precision.as_fraction() == Fraction(5, 6)


def test_incompatible_local_parents_are_rejected() -> None:
    left = window(((Fraction(1, 2), 1),))
    right = window(((Fraction(1, 2), 1),), center=Fraction(1))
    with pytest.raises(
        OperationDomainValidationError, match="share variable and center"
    ):
        add_puiseux(left, right)


def test_forged_window_rejects_variable_outside_polynomial_identifier_grammar() -> None:
    valid = window(((Fraction(1, 2), 1),))
    forged = TruncatedPuiseuxWindow.model_construct(
        variable="bad-name",
        center=valid.center,
        valuation_lower=valid.valuation_lower,
        precision=valid.precision,
        ramification_index=valid.ramification_index,
        terms=valid.terms,
    )

    with pytest.raises(
        OperationDomainValidationError, match="polynomial identifier grammar"
    ):
        add_puiseux(forged, forged)


def test_product_rejects_coefficient_growth_before_convolution() -> None:
    huge = 10**3_000
    left = window(((Fraction(0), huge),), precision=Fraction(1))
    right = window(((Fraction(0), huge),), precision=Fraction(1))

    with pytest.raises(OperationResourceAdmissionError, match="coefficient growth"):
        multiply_puiseux(left, right)


def test_product_rejects_pair_work_before_convolution() -> None:
    dense_terms = tuple((Fraction(i), 1) for i in range(1_001))
    left = window(dense_terms, precision=Fraction(1_002))

    with pytest.raises(OperationResourceAdmissionError, match="convolution work"):
        multiply_puiseux(left, left)


def test_add_result_byte_preflight_accepts_at_boundary_and_rejects_above() -> None:
    left_at_boundary = window(
        tuple((Fraction(2 * i), 1) for i in range(619)),
        precision=Fraction(2_000),
    )
    right_at_boundary = window(
        tuple((Fraction(2 * i + 1), 1) for i in range(619)),
        precision=Fraction(2_000),
    )
    assert len(add_puiseux(left_at_boundary, right_at_boundary).terms) == 1_238

    left_over = window(
        tuple((Fraction(2 * i), 1) for i in range(620)),
        precision=Fraction(2_000),
    )
    right_over = right_at_boundary
    with pytest.raises(
        OperationResourceAdmissionError, match="serialized-size envelope"
    ):
        add_puiseux(left_over, right_over)


def test_single_near_limit_rational_is_copied_by_add_and_product_by_one() -> None:
    numerator = 10**4_095 + 1
    denominator = 10**4_095 + 3
    assert gcd(numerator, denominator) == 1
    coefficient = q(numerator, denominator)
    assert len(str(numerator)) == 4_096
    assert len(str(denominator)) == 4_096
    source = TruncatedPuiseuxWindow(
        valuation_lower=q(0),
        precision=q(1),
        ramification_index=1,
        terms=(PuiseuxTerm(exponent=q(0), coefficient=coefficient),),
    )
    empty = TruncatedPuiseuxWindow(
        valuation_lower=q(0), precision=q(1), ramification_index=1, terms=()
    )
    one = TruncatedPuiseuxWindow(
        valuation_lower=q(0),
        precision=q(1),
        ramification_index=1,
        terms=(PuiseuxTerm(exponent=q(0), coefficient=q(1)),),
    )

    assert add_puiseux(source, empty) == source
    assert multiply_puiseux(source, one) == source
