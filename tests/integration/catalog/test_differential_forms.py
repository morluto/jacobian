"""Public-dispatch composition for polynomial differential-form wedge."""

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
