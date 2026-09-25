from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    ThetaSelectedCoefficientsRequest,
    ThetaSeriesPrefixRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_selected_coefficients,
    theta_series_prefix,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(
    diagonal: tuple[int, ...], crosses: tuple[tuple[int, int, int], ...] = ()
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(len(diagonal))),
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
        axis=("x",), diagonal_coefficients=(CanonicalRational(num=1, den=2),)
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


def test_selected_theta_coefficients_match_direct_finite_enumeration() -> None:
    form = _form((1,))
    # Directly enumerate every x with x^2 <= 10^6. No theta-prefix kernel or
    # shared coordinate-bound helper is used by this oracle.
    expected = {0: 0, 17: 0, 1_000_000: 0}
    for (x,) in product(range(-1000, 1001), repeat=1):
        value = x * x
        if value in expected:
            expected[value] += 1
    result = theta_selected_coefficients(form, (0, 17, 1_000_000))
    assert tuple((row.index, row.coefficient) for row in result.coefficients) == tuple(
        expected.items()
    )
    assert result.form == form


def test_selected_theta_coefficients_match_cross_term_oracle_and_roundtrip() -> None:
    from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
        ThetaSelectedCoefficientsResult,
    )

    form = _form((1, 1), ((0, 1, 1),))
    indices = (0, 1, 2, 3, 6, 10)
    expected = dict.fromkeys(indices, 0)
    for vector in product(range(-4, 5), repeat=2):
        value = vector[0] ** 2 + vector[0] * vector[1] + vector[1] ** 2
        if value in expected:
            expected[value] += 1
    result = theta_selected_coefficients(form, indices)
    assert tuple((row.index, row.coefficient) for row in result.coefficients) == tuple(
        (index, expected[index]) for index in indices
    )
    assert (
        ThetaSelectedCoefficientsResult.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize("indices", [(1, 1), (2, 1), (-1,), (1_000_000_001,)])
def test_selected_theta_request_requires_canonical_bounded_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        ThetaSelectedCoefficientsRequest(form=_form((1,)), indices=indices)


def test_selected_theta_native_boundary_revalidates_nested_form_and_schema() -> None:
    form = _form((1,))
    forged_form = form.model_copy(
        update={
            "cross_terms": (
                QuadraticCrossTerm.model_construct(
                    left=0, right=4, coefficient=CanonicalRational(num=1, den=1)
                ),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        theta_selected_coefficients(forged_form, (0,))
    assert exc_info.value.errors()[0]["type"] == "quadratic_form.theta_invalid_form"

    schema = ThetaSelectedCoefficientsRequest.model_json_schema()
    item = schema["properties"]["indices"]["items"]
    assert item["type"] == "integer"
    assert item["minimum"] == 0
    assert item["maximum"] == 1_000_000_000


def test_selected_theta_admission_bounds_proved_search_box() -> None:
    form = _form((1, 1, 1, 1, 1, 1, 1))
    with pytest.raises(
        OperationResourceAdmissionError, match="lattice box"
    ) as exc_info:
        theta_selected_coefficients(form, (1_000_000,))
    assert exc_info.value.errors()[0]["loc"] == ("indices",)


def test_selected_theta_operation_composes_through_json_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation_id = "quadratic_form.theta_selected_coefficients.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    request = {
        "form": {
            "axis": ["x"],
            "diagonal_coefficients": [{"num": "1", "den": "1"}],
        },
        "indices": [0, 17, 1_000_000],
    }
    operation.request_type.model_validate_json(encode_strict_json(request), strict=True)
    response = invoke_operation(operation_id, request, catalog)
    assert response.output["coefficients"] == [
        {"index": 0, "coefficient": 1},
        {"index": 17, "coefficient": 0},
        {"index": 1_000_000, "coefficient": 2},
    ]
    decoded = operation.result_type.model_validate_json(
        encode_strict_json(response.output), strict=True
    )
    assert decoded.model_dump(mode="json") == response.output
