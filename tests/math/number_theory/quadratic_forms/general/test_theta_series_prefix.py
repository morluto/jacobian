from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    ThetaSeriesPrefixRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_series_prefix,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(diagonal: tuple[int, ...], crosses: tuple[tuple[int, int, int], ...] = ()):
    return RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(len(diagonal))),
        diagonal_coefficients=tuple({"num": value, "den": 1} for value in diagonal),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left,
                right=right,
                coefficient={"num": value, "den": 1},
            )
            for left, right, value in crosses
        ),
    )


def _brute_prefix(
    form: RationalQuadraticForm, cutoff: int, radius: int
) -> tuple[int, ...]:
    result = [0] * (cutoff + 1)
    for vector in product(range(-radius, radius + 1), repeat=len(form.axis)):
        value = sum(
            coefficient.num * coordinate * coordinate
            for coefficient, coordinate in zip(
                form.diagonal_coefficients, vector, strict=True
            )
        )
        value += sum(
            term.coefficient.num * vector[term.left] * vector[term.right]
            for term in form.cross_terms
        )
        if 0 <= value <= cutoff:
            result[value] += 1
    return tuple(result)


def test_theta_prefix_matches_independent_box_oracle_with_cross_term() -> None:
    form = _form((1, 1), ((0, 1, 1),))
    result = theta_series_prefix(ThetaSeriesPrefixRequest(form=form, cutoff=12))
    assert result.coefficients == _brute_prefix(form, 12, 4)
    assert result.coefficients[:5] == (1, 6, 0, 6, 6)
    assert result.form == form
    assert result.cutoff == 12


def test_theta_prefix_admits_zero_dimensional_positive_definite_form() -> None:
    form = _form(())
    result = theta_series_prefix(ThetaSeriesPrefixRequest(form=form, cutoff=4))
    assert result.coefficients == (1, 0, 0, 0, 0)


def test_theta_prefix_rejects_nonintegral_and_indefinite_forms() -> None:
    nonintegral = RationalQuadraticForm(
        axis=("x",), diagonal_coefficients=({"num": 1, "den": 2},)
    )
    with pytest.raises(OperationDomainValidationError, match="integral"):
        theta_series_prefix(ThetaSeriesPrefixRequest(form=nonintegral, cutoff=1))
    with pytest.raises(OperationDomainValidationError, match="positive-definite"):
        theta_series_prefix(ThetaSeriesPrefixRequest(form=_form((1, -1)), cutoff=3))


def test_theta_prefix_rejects_nonpositive_definite_singular_form() -> None:
    with pytest.raises(OperationDomainValidationError, match="positive-definite"):
        theta_series_prefix(
            ThetaSeriesPrefixRequest(form=_form((1, 1), ((0, 1, 2),)), cutoff=3)
        )


def test_theta_prefix_rejects_when_proved_box_exceeds_vector_admission() -> None:
    broad = _form((1, 1, 1, 1, 1, 1, 1))
    with pytest.raises(OperationResourceAdmissionError, match="lattice box"):
        theta_series_prefix(ThetaSeriesPrefixRequest(form=broad, cutoff=512))


def test_theta_prefix_result_requires_complete_prefix_shape() -> None:
    from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
        ThetaSeriesPrefixResult,
    )

    with pytest.raises(ValidationError, match=r"q\^0 through q\^cutoff"):
        ThetaSeriesPrefixResult(form=_form((1,)), cutoff=2, coefficients=(1, 2))
