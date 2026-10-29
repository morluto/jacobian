"""Strict wire sources and native zero shortcuts agree on canonical axes."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.polynomials.differential_forms.operations import exterior_derivative
from jacobian.math.polynomials.differential_forms.values import (
    PolynomialDifferentialForm,
)


@pytest.mark.parametrize("axis", (("x", "x"), ("!",), ("x" * 33,), (1,)))
def test_invalid_zero_form_axes_reject_at_native_and_wire_boundaries(
    axis: tuple[object, ...],
) -> None:
    source = PolynomialDifferentialForm.model_construct(
        variables=axis, degree=0, components=()
    )
    with pytest.raises(OperationDomainValidationError) as native_error:
        exterior_derivative(source)
    assert native_error.value.errors()[0]["type"] == "differential_form.variable_axis"
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            "differential_form.exterior_derivative.compute",
            {"form": {"variables": list(axis), "degree": "0", "components": []}},
            Catalog.open(),
        )


def test_public_overdimensional_zero_derivative_composes_unchanged() -> None:
    catalog = Catalog.open()
    source = {"variables": ["Y_2", "x"], "degree": "3", "components": []}
    first = invoke_operation(
        "differential_form.exterior_derivative.compute", {"form": source}, catalog
    )
    decoded = PolynomialDifferentialForm.model_validate_json(json.dumps(first.output))
    assert decoded.variables == ("Y_2", "x")
    assert decoded.degree == 4
    assert decoded.components == ()
    second = invoke_operation(
        "differential_form.exterior_derivative.compute",
        {"form": decoded.model_dump(mode="json")},
        catalog,
    )
    assert (
        PolynomialDifferentialForm.model_validate_json(json.dumps(second.output)).degree
        == 5
    )
