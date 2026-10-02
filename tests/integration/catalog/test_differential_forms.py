"""Public-dispatch composition for polynomial differential forms."""

from typing import Any

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_catalog_round_trip_preserves_degree_labelled_zero() -> None:
    catalog = Catalog.open()
    operation_id = "differential_form.wedge.compute"
    top = {
        "variables": ["x", "y"],
        "degree": "2",
        "components": [],
    }
    result = invoke_operation(
        operation_id,
        {"left": top, "right": top},
        catalog,
    )
    assert result.output == {
        "variables": ["x", "y"],
        "degree": "4",
        "components": [],
    }

    operation = catalog.operation(operation_id)
    assert operation is not None
    round_tripped = operation.result_type.model_validate_json(
        encode_strict_json(result.output)
    )
    assert round_tripped.model_dump(mode="json") == result.output

    scalar_zero = {
        "variables": ["x", "y"],
        "degree": "0",
        "components": [],
    }
    composed = invoke_operation(
        operation_id,
        {"left": result.output, "right": scalar_zero},
        catalog,
    )
    assert composed.output == result.output


def test_catalog_round_trip_preserves_signed_nonzero_wedge() -> None:
    catalog = Catalog.open()
    operation_id = "differential_form.wedge.compute"
    coefficient = {
        "variables": ["x", "y", "z"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [0, 0, 0],
                }
            ]
        },
    }
    result = invoke_operation(
        operation_id,
        {
            "left": {
                "variables": ["x", "y", "z"],
                "degree": "2",
                "components": [{"indices": [0, 2], "coefficient": coefficient}],
            },
            "right": {
                "variables": ["x", "y", "z"],
                "degree": "1",
                "components": [{"indices": [1], "coefficient": coefficient}],
            },
        },
        catalog,
    )
    assert result.output["components"][0]["indices"] == [0, 1, 2]
    assert result.output["components"][0]["coefficient"]["polynomial"]["terms"][0][
        "coefficient"
    ] == {"num": "-1", "den": "1"}
    operation = catalog.operation(operation_id)
    assert operation is not None
    round_tripped = operation.result_type.model_validate_json(
        encode_strict_json(result.output)
    )
    assert round_tripped.model_dump(mode="json") == result.output


def test_scalar_lie_derivatives_compose_through_serialized_results() -> None:
    catalog = Catalog.open()
    operation_id = "differential_form.lie_derivative.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None

    def scalar(coefficient: int, power: int) -> dict[str, Any]:
        return {
            "variables": ["x", "unused"],
            "degree": "0",
            "components": [
                {
                    "indices": [],
                    "coefficient": {
                        "domain": "QQ",
                        "variables": ["x", "unused"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {
                                        "num": str(coefficient),
                                        "den": "1",
                                    },
                                    "exponents": [power, 0],
                                }
                            ],
                        },
                    },
                },
            ]
            if coefficient
            else [],
        }

    field = {
        "variables": ["x", "unused"],
        "components": [
            scalar(1, 0)["components"][0]["coefficient"],
            {"variables": ["x", "unused"], "polynomial": {"terms": []}},
        ],
    }
    current = scalar(1, 2)
    # Repeated partial_x derivatives: x^2 -> 2x -> 2 -> 0 -> 0.
    for expected in (scalar(2, 1), scalar(2, 0), scalar(0, 0), scalar(0, 0)):
        result = invoke_operation(
            operation_id, {"field": field, "form": current}, catalog
        )
        assert result.output == expected
        restored = operation.result_type.model_validate_json(
            encode_strict_json(result.output)
        )
        assert restored.model_dump(mode="json") == expected
        current = result.output


def test_empty_axis_scalar_lie_result_is_consumable() -> None:
    catalog = Catalog.open()
    scalar_zero = {"variables": [], "degree": "0", "components": []}
    result = invoke_operation(
        "differential_form.lie_derivative.compute",
        {"field": {"variables": [], "components": []}, "form": scalar_zero},
        catalog,
    )
    assert result.output == scalar_zero
    composed = invoke_operation(
        "differential_form.wedge.compute",
        {"left": result.output, "right": result.output},
        catalog,
    )
    assert composed.output == scalar_zero
