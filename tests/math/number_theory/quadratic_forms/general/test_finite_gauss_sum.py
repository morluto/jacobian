from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicField,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FiniteGaussSumRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.extra_operations import (
    finite_quadratic_gauss_sum,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(
    axis: tuple[str, ...], diagonal: tuple[int, ...], cross=()
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(
            CanonicalRational(num=value, den=1) for value in diagonal
        ),
        cross_terms=tuple(cross),
    )


def test_mod_five_sum_matches_independent_classical_gauss_sum() -> None:
    # For the Legendre symbol modulo 5, sum_x zeta_5^(x^2)=sqrt(5).
    # Since sqrt(5)=1+2(zeta_5+zeta_5^-1), reduce with
    # Phi_5(z)=1+z+z^2+z^3+z^4 to obtain -1-2z^2-2z^3.
    result = finite_quadratic_gauss_sum(
        FiniteGaussSumRequest(form=_form(("x",), (1,)), modulus=5)
    )
    assert result.histogram == (1, 2, 0, 0, 2)
    assert result.total == 5
    assert result.value.field == RationalCyclotomicField(order=5)
    assert tuple(
        (coefficient.num, coefficient.den)
        for coefficient in result.value.coefficients_ascending
    ) == ((-1, 1), (0, 1), (-2, 1), (-2, 1))


def test_mixed_form_profile_matches_direct_residue_oracle() -> None:
    form = _form(
        ("x", "y"),
        (1, 2),
        (
            QuadraticCrossTerm(
                left=0, right=1, coefficient=CanonicalRational(num=3, den=1)
            ),
        ),
    )
    result = finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=4))

    # Independent direct enumeration of Q(x,y)=x^2+3xy+2y^2 modulo 4.
    expected = [0] * 4
    for x in range(4):
        for y in range(4):
            expected[(x * x + 3 * x * y + 2 * y * y) % 4] += 1
    assert list(result.histogram) == expected
    assert result.total == 16 == sum(expected)
    assert result.value.field == RationalCyclotomicField(order=4)


def test_zero_dimensional_form_has_one_term_for_every_modulus() -> None:
    result = finite_quadratic_gauss_sum(
        FiniteGaussSumRequest(form=_form((), ()), modulus=7)
    )
    assert result.total == 1
    assert result.histogram == (1, 0, 0, 0, 0, 0, 0)
    assert tuple(v.num for v in result.value.coefficients_ascending) == (
        1,
        0,
        0,
        0,
        0,
        0,
    )


def test_rejects_residue_space_above_bound_before_enumeration() -> None:
    request = FiniteGaussSumRequest(
        form=_form(("x", "y", "z", "w"), (1, 1, 1, 1)), modulus=64
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="complete residue domain"
    ):
        finite_quadratic_gauss_sum(request)


def test_rejects_dense_support_before_enumeration() -> None:
    # A dense 20-variable form with modulus 2 has only 2^20 states, but each
    # state evaluates all 210 polynomial terms.
    axis = tuple(f"x{index}" for index in range(20))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(CanonicalRational(num=1, den=1) for _ in axis),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left,
                right=right,
                coefficient=CanonicalRational(num=1, den=1),
            )
            for left in range(20)
            for right in range(left + 1, 20)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=2))


def test_rejects_oversized_support_before_enumeration() -> None:
    # Modulus 1 always has a single state, so only the support bound guards
    # the retained source and per-state traversal.
    axis = tuple(f"x{index}" for index in range(100))
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(CanonicalRational(num=1, den=1) for _ in axis),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left,
                right=right,
                coefficient=CanonicalRational(num=1, den=1),
            )
            for left in range(100)
            for right in range(left + 1, 100)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="support"):
        finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=1))


def test_rejects_retained_source_above_output_digit_envelope() -> None:
    # Modulus 1 always enumerates a single residue state, so the state and
    # work bounds admit this schema-valid form, but the result retains the
    # complete 2080-term form with 256-digit coefficients: well over one
    # megabyte of canonical decimal digits for a one-state computation.
    axis = tuple(f"x{index}" for index in range(64))
    tall = 10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(CanonicalRational(num=tall, den=1) for _ in axis),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left,
                right=right,
                coefficient=CanonicalRational(num=tall, den=1),
            )
            for left in range(64)
            for right in range(left + 1, 64)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError, match="output digit envelope"):
        finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=1))


def test_small_support_with_tall_coefficients_remains_admitted() -> None:
    # The output envelope must not evict tall-but-small forms: 10^256-1 is
    # divisible by 3, so every residue of Q(x) = tall*x^2 vanishes mod 3.
    tall = 10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1
    result = finite_quadratic_gauss_sum(
        FiniteGaussSumRequest(form=_form(("x",), (tall,)), modulus=3)
    )
    assert result.histogram == (3, 0, 0)
    assert result.total == 3


def test_rejects_rational_coefficients() -> None:
    form = RationalQuadraticForm(
        axis=("x",),
        diagonal_coefficients=(CanonicalRational(num=1, den=2),),
    )
    with pytest.raises(ValueError, match="integral coefficients"):
        finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=3))
