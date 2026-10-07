"""Native zero shortcuts retain only canonical form axes and degrees."""

from collections.abc import Callable

import pytest

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms.operations import (
    affine_homotopy_primitive,
    exterior_derivative,
    interior_product,
    lie_derivative,
    pullback,
    wedge,
)
from jacobian.math.polynomials.differential_forms.values import (
    PolynomialDifferentialForm,
    PolynomialMap,
    PolynomialVectorField,
)

_EMPTY = PolynomialDifferentialForm(variables=(), degree=0)
_FIELD = PolynomialVectorField(variables=(), components=())
_MAP = PolynomialMap(source_variables=(), target_variables=(), images=())
_PRODUCERS: tuple[Callable[[PolynomialDifferentialForm], StrictModel], ...] = (
    exterior_derivative,
    affine_homotopy_primitive,
    lambda form: wedge(form, _EMPTY),
    lambda form: interior_product(_FIELD, form),
    lambda form: lie_derivative(_FIELD, form),
    lambda form: pullback(_MAP, form),
)


@pytest.mark.parametrize("producer", _PRODUCERS)
@pytest.mark.parametrize("axis", (("x", "x"), ("!",), ("x" * 33,), (1,), (["x"],)))
def test_zero_form_native_admission_rejects_invalid_axis(
    producer: Callable[[PolynomialDifferentialForm], StrictModel], axis: object
) -> None:
    forged = _EMPTY.model_copy(update={"variables": axis})
    with pytest.raises(OperationDomainValidationError) as error:
        producer(forged)
    assert error.value.errors()[0]["type"] == "differential_form.variable_axis"


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_zero_form_native_admission_rejects_boolean_degree(
    producer: Callable[[PolynomialDifferentialForm], StrictModel],
) -> None:
    forged = _EMPTY.model_copy(update={"degree": True})
    with pytest.raises(OperationDomainValidationError) as error:
        producer(forged)
    assert error.value.errors()[0]["type"] == "differential_form.degree"


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_zero_form_native_admission_bounds_input_degree(
    producer: Callable[[PolynomialDifferentialForm], StrictModel],
) -> None:
    forged = _EMPTY.model_copy(update={"degree": 10**MAX_CANONICAL_INTEGER_DIGITS})
    with pytest.raises(OperationResourceAdmissionError) as error:
        producer(forged)
    assert error.value.errors()[0]["type"] == "differential_form.wedge.degree_budget"


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_empty_axis_zero_shortcuts_round_trip(
    producer: Callable[[PolynomialDifferentialForm], StrictModel],
) -> None:
    result = producer(_EMPTY)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_overdimensional_zero_retains_valid_ordered_axis_and_grade() -> None:
    source = PolynomialDifferentialForm(variables=("Y_2", "x"), degree=3)
    derivative = exterior_derivative(source)
    assert derivative.variables == source.variables
    assert derivative.degree == 4
    assert derivative.components == ()
    decoded = PolynomialDifferentialForm.model_validate_json(
        derivative.model_dump_json()
    )
    assert exterior_derivative(decoded).degree == 5
