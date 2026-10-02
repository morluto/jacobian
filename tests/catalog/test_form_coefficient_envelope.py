"""Discovery states the enforced widened form carrier."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
)


def test_form_producer_inspection_publishes_the_same_carrier() -> None:
    assert MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS == 8192
    catalog = Catalog.open()
    for name in (
        "contract",
        "pullback",
        "primitive",
        "lie_derivative",
        "exterior_derivative",
    ):
        operation = catalog.operation(f"differential_form.{name}.compute")
        assert operation is not None
        request = operation.request_type.model_json_schema()
        assert "8192" in request["properties"]["form"]["description"]
        result = operation.result_type.model_json_schema()
        assert (
            "8192"
            in result["$defs"]["FormComponent"]["properties"]["coefficient"][
                "description"
            ]
        )
