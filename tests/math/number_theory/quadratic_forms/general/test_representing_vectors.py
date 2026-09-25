from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    ThetaRepresentingVectorsRequest,
    ThetaRepresentingVectorsResult,
    ThetaRepresentingVectorsRow,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_representing_vectors,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(
    diagonal: tuple[int, ...], crosses: tuple[tuple[int, int, int], ...] = ()
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=tuple(f"x{index}" for index in range(len(diagonal))),
        diagonal_coefficients=tuple(
            CanonicalRational(num=value, den=1) for value in diagonal
        ),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=left,
                right=right,
                coefficient=CanonicalRational(num=value, den=1),
            )
            for left, right, value in crosses
        ),
    )


def _direct_vectors(
    diagonal: tuple[int, ...],
    crosses: tuple[tuple[int, int, int], ...],
    indices: tuple[int, ...],
    radius: int,
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    table: dict[int, list[tuple[int, ...]]] = {index: [] for index in indices}
    for vector in product(range(-radius, radius + 1), repeat=len(diagonal)):
        value = sum(
            coefficient * coordinate**2
            for coefficient, coordinate in zip(diagonal, vector, strict=True)
        )
        value += sum(
            coefficient * vector[left] * vector[right]
            for left, right, coefficient in crosses
        )
        if value in table:
            table[value].append(tuple(vector))
    return tuple((index, tuple(table[index])) for index in indices)


def test_representing_vectors_match_direct_sum_of_two_squares_oracle() -> None:
    diagonal = (1, 1)
    indices = (0, 1, 2, 3, 4, 5)
    expected = _direct_vectors(diagonal, (), indices, radius=2)
    request = ThetaRepresentingVectorsRequest(form=_form(diagonal), indices=indices)
    result = theta_representing_vectors(request)
    actual = tuple(
        (row.index, tuple(tuple(vector) for vector in row.vectors))
        for row in result.rows
    )
    assert actual == expected
    assert result.rows[3].vectors == ()
    assert result.rows[2].vectors == (
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    assert result.model_dump(mode="json")["rows"][2]["vectors"][0] == ["-1", "-1"]
    assert (
        ThetaRepresentingVectorsResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_representing_vectors_preserve_cross_term_signs_and_order() -> None:
    diagonal = (1, 1)
    crosses = ((0, 1, 1),)
    indices = (0, 1, 2, 3)
    expected = _direct_vectors(diagonal, crosses, indices, radius=2)
    result = theta_representing_vectors(
        ThetaRepresentingVectorsRequest(form=_form(diagonal, crosses), indices=indices)
    )
    assert (
        tuple(
            (row.index, tuple(tuple(vector) for vector in row.vectors))
            for row in result.rows
        )
        == expected
    )


def test_zero_dimensional_form_has_one_empty_vector_at_value_zero() -> None:
    result = theta_representing_vectors(
        ThetaRepresentingVectorsRequest(form=_form(()), indices=(0, 1))
    )
    assert result.rows[0].vectors == ((),)
    assert result.rows[1].vectors == ()


@pytest.mark.parametrize("indices", [(2, 1), (1, 1), (-1,), (1_000_000_001,)])
def test_representation_request_requires_canonical_bounded_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        ThetaRepresentingVectorsRequest(form=_form((1,)), indices=indices)


def test_representing_vectors_reject_nonintegral_and_nonpositive_forms() -> None:
    nonintegral = RationalQuadraticForm(
        axis=("x",),
        diagonal_coefficients=(CanonicalRational(num=1, den=2),),
    )
    with pytest.raises(OperationDomainValidationError, match="integral"):
        theta_representing_vectors(
            ThetaRepresentingVectorsRequest(form=nonintegral, indices=(0,))
        )
    with pytest.raises(OperationDomainValidationError, match="positive-definite"):
        theta_representing_vectors(
            ThetaRepresentingVectorsRequest(form=_form((1, -1)), indices=(2,))
        )


def test_representing_vectors_reject_unbounded_proved_search_box() -> None:
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        theta_representing_vectors(
            ThetaRepresentingVectorsRequest(
                form=_form((1, 1, 1, 1, 1, 1, 1)), indices=(1_000_000,)
            )
        )
    assert exc_info.value.errors()[0]["loc"] == ("indices",)


def test_representation_result_enforces_axis_shape_order_and_coordinate_bound() -> None:
    form = _form((1,))
    with pytest.raises(ValidationError, match="length must match form axis"):
        ThetaRepresentingVectorsResult(
            form=form,
            rows=(ThetaRepresentingVectorsRow(index=1, vectors=((1, 0),)),),
        )
    with pytest.raises(ValidationError, match="unique and ordered"):
        ThetaRepresentingVectorsResult(
            form=form,
            rows=(ThetaRepresentingVectorsRow(index=1, vectors=((1,), (1,))),),
        )
    with pytest.raises(ValidationError, match="coordinate exceeds its bound"):
        ThetaRepresentingVectorsResult(
            form=form,
            rows=(ThetaRepresentingVectorsRow(index=1, vectors=((50_000,),)),),
        )
