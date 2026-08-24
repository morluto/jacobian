"""Public native API tests for integral binary quadratic forms."""

from jacobian.math import integral_binary_quadratic_forms
from jacobian.math.integral_binary_quadratic_forms._models import (
    BinaryQuadraticFormCheckRequest,
)
from jacobian.math.integral_binary_quadratic_forms._operations import compute_check


def test_public_form_value_and_native_functions_compose_after_serialization() -> None:
    checked = compute_check(BinaryQuadraticFormCheckRequest(a=5, b=3, c=1))
    assert checked.form is not None

    form = integral_binary_quadratic_forms.PrimitivePositiveDefiniteBinaryQuadraticForm.model_validate(
        checked.model_dump(mode="json")["form"]
    )
    assert integral_binary_quadratic_forms.evaluate(form, 1, 0) == 5
    assert integral_binary_quadratic_forms.reduced_form(form).model_dump() == {
        "a": 1,
        "b": 1,
        "c": 3,
    }
    rows = integral_binary_quadratic_forms.representations(form, 5)
    assert len(rows) == 4
    assert all(
        isinstance(
            row, integral_binary_quadratic_forms.BinaryQuadraticFormRepresentation
        )
        for row in rows
    )
