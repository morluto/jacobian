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


def test_rejects_rational_coefficients() -> None:
    form = RationalQuadraticForm(
        axis=("x",),
        diagonal_coefficients=(CanonicalRational(num=1, den=2),),
    )
    with pytest.raises(ValueError, match="integral coefficients"):
        finite_quadratic_gauss_sum(FiniteGaussSumRequest(form=form, modulus=3))
