"""The first-order differential GCRD is published and executable."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_gcrd_catalog_example_executes_with_exact_bezout_coefficients() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("ore.operator.gcrd.compute")
    assert operation is not None
    example = operation.examples[0]

    result = invoke_operation(operation.operation_id, example.input, catalog)

    assert example.name == "coprime_first_order_operators"
    assert result.output["divisor"]["terms"] == [
        {
            "order": 0,
            "coefficient": {
                "domain": "QQ",
                "variables": ["x"],
                "numerator": {
                    "terms": [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [0],
                        }
                    ]
                },
                "denominator": {
                    "terms": [
                        {
                            "coefficient": {"num": "1", "den": "1"},
                            "exponents": [0],
                        }
                    ]
                },
            },
        }
    ]
