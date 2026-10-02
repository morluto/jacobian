"""Public vector derivatives preserve typed coefficient resource refusals."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation


@pytest.mark.parametrize("operation", ["divergence", "curl"])
@pytest.mark.parametrize("kind", ["output", "work"])
def test_public_vector_derivative_resource_admission(operation: str, kind: str) -> None:
    if kind == "output":
        coefficient = "9" * 32768
        exponents = [(2, 0, 0) if operation == "divergence" else (0, 2, 0)]
        code = "polynomial_vector_calc.derivative_coefficient_bound"
    else:
        coefficient = "1" + "0" * 23000
        exponents = [(2, 0, 0), (1, 0, 0), (0, 0, 0)]
        code = "polynomial_vector_calc.coefficient_work_budget"
    components = [
        {
            "variables": ["x", "y", "z"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": "1"},
                        "exponents": list(powers),
                    }
                    for powers in exponents
                ]
            },
        },
        {"variables": ["x", "y", "z"], "polynomial": {"terms": []}},
        {"variables": ["x", "y", "z"], "polynomial": {"terms": []}},
    ]
    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            f"polynomial_field.vector.{operation}.compute",
            {"components": components},
            Catalog.open(),
        )
    assert error.value.errors()[0]["type"] == code
    assert error.value.errors()[0]["loc"] == ("components",)
